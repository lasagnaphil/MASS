import argparse
import gym
import numpy as np

import ray
from ray import tune
from ray.tune.logger import pretty_print
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer, DDPPOTrainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ars import ARSTrainer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env

torch, nn = try_import_torch()

from pymss import SingleEnvManager
try:
    from pymss import MultiEnvManager
except ImportError:
    MultiEnvManager = None

from Model import *
import mcmc

from typing import Dict, List, Callable
from collections import namedtuple

import os
import math
import random
import itertools

MuscleTransition = namedtuple('MuscleTransition', ('JtA', 'tau_des', 'L', 'b'))
MarginalTransition = namedtuple('MarginalTransition', ('sb', 'v'))

class MyEnv(gym.Env):
    def __init__(self, config):
        self.env = SingleEnvManager(config)

        self.use_muscle = self.env.UseMuscle()
        self.use_adaptive_sampling = self.env.UseAdaptiveSampling()

        self.num_state = self.env.GetNumState()
        self.num_action = self.env.GetNumAction()
        if self.use_muscle:
            self.num_muscles = self.env.GetNumMuscles()
            self.num_muscle_dofs = self.env.GetNumTotalMuscleRelatedDofs()
        if self.use_adaptive_sampling:
            self.num_paramstate = self.env.GetNumParams()

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_state,))
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_action,))

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.use_muscle:
            self.muscle_model = MuscleNN(self.num_muscle_dofs, self.num_action, self.num_muscles).to(
                self.device)

        self.num_simulation_Hz = self.env.GetSimulationHz()
        self.num_control_Hz = self.env.GetControlHz()
        self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

        if self.use_adaptive_sampling:
            self.marginal_tuples = []
            self.initial_param_states = []

        self.counter = 0
        self.num_transitions_so_far = 0
        self.stats = {}

    def reset(self):
        if self.use_adaptive_sampling:
            initial_state = np.float32(random.choice(self.initial_param_states))
            self.env.ResetWithParams(True, initial_state)
        else:
            self.env.Reset(True)
        return self.env.GetState()

    def step(self, action):
        self.env.SetAction(action)
        if self.use_muscle:
            mt = torch.from_numpy(self.env.GetMuscleTorques()).to(self.device)
            for _ in range(self.num_simulation_per_control // 2):
                dt = torch.from_numpy(self.env.GetDesiredTorques()).to(self.device)
                activations = self.muscle_model(mt, dt).cpu().detach().numpy()
                self.env.SetActivationLevels(activations)
                self.env.Steps(2)
        else:
            self.env.StepsAtOnce()

        self.counter += 1

        obs = self.env.GetState()
        reward = self.env.GetReward()
        done = self.env.IsEndOfEpisode() or np.isnan(reward)
        info = {}

        if self.use_adaptive_sampling:
            self.marginal_tuples.append(obs[-self.num_paramstate:])

        return obs, reward, done, info

    def load_muscle_model_weights(self, weights):
        assert(self.use_muscle)
        self.muscle_model.load_state_dict(weights)

    def get_muscle_tuples(self):
        assert(self.use_muscle)
        muscle_tuples = self.env.GetMuscleTuples()
        return [MuscleTransition(t[0], t[1], t[2], t[3]) for t in muscle_tuples]

    def set_initial_param_states(self, initial_param_states):
        assert(self.use_adaptive_sampling)
        self.initial_param_states = initial_param_states

    def get_marginal_tuples(self):
        assert(self.use_adaptive_sampling)
        return self.marginal_tuples

    def clear_marginal_tuples(self):
        assert(self.use_adaptive_sampling)
        self.marginal_tuples.clear()

class MyVectorEnv(VectorEnv):
    def __init__(self, config):
        self.env = MultiEnvManager(config)

        self.use_muscle = self.env.UseMuscle()
        self.use_adaptive_sampling = self.env.UseAdaptiveSampling()

        self.num_state = self.env.GetNumState()
        self.num_action = self.env.GetNumAction()
        if self.use_muscle:
            self.num_muscles = self.env.GetNumMuscles()
            self.num_muscle_dofs = self.env.GetNumTotalMuscleRelatedDofs()
        if self.use_adaptive_sampling:
            self.num_paramstate = self.env.GetNumParams()

        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_state,))
        action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_action,))
        super(MyVectorEnv, self).__init__(obs_space, action_space, config["num_envs"])

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.use_muscle:
            self.muscle_model = MuscleNN(self.num_muscle_dofs, self.num_action, self.num_muscles).to(
                self.device)

        self.num_simulation_Hz = self.env.GetSimulationHz()
        self.num_control_Hz = self.env.GetControlHz()
        self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

        if self.use_adaptive_sampling:
            self.marginal_tuples = []
            self.initial_param_states = []

        self.counter = 0
        self.num_transitions_so_far = 0
        self.stats = {}

    @override(VectorEnv)
    def vector_reset(self):
        if self.use_adaptive_sampling:
            for j in range(self.num_envs):
                initial_state = np.float32(random.choice(self.initial_param_states))
                self.env.ResetWithParams(True, j, initial_state)
        else:
            for j in range(self.num_envs):
                self.env.Reset(True, j)
        return self.env.GetStates()

    @override(VectorEnv)
    def reset_at(self, index: int):
        if self.use_adaptive_sampling:
            initial_state = np.float32(random.choice(self.initial_param_states))
            self.env.ResetWithParams(True, index, initial_state)
        else:
            self.env.Reset(True, index)
        return self.env.GetState(index)

    @override(VectorEnv)
    def vector_step(self, actions):
        self.env.SetActions(actions)
        if self.use_muscle:
            mt = torch.from_numpy(self.env.GetMuscleTorques()).to(self.device)
            for _ in range(self.num_simulation_per_control // 2):
                dt = torch.from_numpy(self.env.GetDesiredTorques()).to(self.device)
                activations = self.muscle_model(mt, dt).cpu().detach().numpy()
                self.env.SetActivationLevels(activations)
                self.env.Steps(2)
        else:
            self.env.StepsAtOnce()

        self.counter += 1

        obs = self.env.GetStates()
        rewards = [0 if np.isnan(reward) else reward for reward in self.env.GetRewards()]
        dones = [self.env.IsEndOfEpisode(i) or np.isnan(rewards[i]) for i in range(self.num_envs)]
        infos = [{} for _ in range(self.num_envs)]

        if self.use_adaptive_sampling:
            for j in range(self.num_envs):
                self.marginal_tuples.append(obs[j, -self.num_paramstate:])

        return obs, rewards, dones, infos

    @override(VectorEnv)
    def get_unwrapped(self):
        return None

    def load_muscle_model_weights(self, weights):
        assert(self.use_muscle)
        self.muscle_model.load_state_dict(weights)

    def get_muscle_tuples(self):
        assert(self.use_muscle)
        muscle_tuples = self.env.GetMuscleTuples()
        return [MuscleTransition(t[0], t[1], t[2], t[3]) for t in muscle_tuples]

    def set_initial_param_states(self, initial_param_states):
        assert(self.use_adaptive_sampling)
        self.initial_param_states = initial_param_states

    def get_marginal_tuples(self):
        assert(self.use_adaptive_sampling)
        return self.marginal_tuples

    def clear_marginal_tuples(self):
        assert(self.use_adaptive_sampling)
        self.marginal_tuples.clear()

class SimulationNN_Ray(SimulationNN, TorchModelV2):
    def __init__(self, obs_space: gym.Space, action_space: gym.Space, config: Dict, *args, **kwargs):
        num_states = np.prod(obs_space.shape)
        num_actions = np.prod(action_space.shape)
        SimulationNN.__init__(self, num_states, num_actions)

        num_outputs = 2 * np.prod(action_space.shape)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, {}, "SimulationNN_Ray")
        self._last_value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        x = obs.reshape(obs.shape[0], -1)
        action_dist, self._last_value = super(SimulationNN_Ray, self).forward(x)
        action_tensor = torch.cat([action_dist.loc, action_dist.scale], dim=1)
        return action_tensor, state

    @override(TorchModelV2)
    def value_function(self):
        return self._last_value.squeeze(1)

class MuscleLearner:
    def __init__(self, num_action, num_muscles, num_muscle_dofs, run_distributed=False, use_ddp=False, rank=0):
        self.num_action = num_action
        self.num_muscles = num_muscles
        self.run_distributed = run_distributed
        self.use_ddp = use_ddp
        self.rank = rank

        self.num_epochs_muscle = 3
        self.muscle_batch_size = 128
        self.default_learning_rate = 1E-4
        self.learning_rate = self.default_learning_rate

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = MuscleNN(num_muscle_dofs, self.num_action, self.num_muscles).to(self.device)
        if self.use_ddp:
            self.ddp_model = DDP(self.model)
            self.model = self.ddp_model.module
            parameters = self.ddp_model.parameters()
        else:
            parameters = self.model.parameters()

        self.optimizer = optim.Adam(parameters, lr=self.learning_rate)
        for param in parameters:
            param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))

        self.stats = {}

    def get_model_weights(self) -> Dict:
        return self.model.state_dict()

    def save(self, name):
        self.model.save(name)

    def learn(self, muscle_transitions: List[MuscleTransition]) -> Dict:
        sum_loss = 0.0
        muscle_transitions = np.array(muscle_transitions)
        for j in range(self.num_epochs_muscle):
            sum_loss = 0.0
            np.random.shuffle(muscle_transitions)
            for i in range(len(muscle_transitions) // self.muscle_batch_size):
                tuples = muscle_transitions[i * self.muscle_batch_size:(i + 1) * self.muscle_batch_size]
                batch = MuscleTransition(*zip(*tuples))

                stack_JtA = torch.from_numpy(np.vstack(batch.JtA).astype(np.float32)).to(self.device)

                stack_tau_des = torch.from_numpy(np.vstack(batch.tau_des).astype(np.float32)).to(self.device)
                stack_L = torch.from_numpy(np.vstack(batch.L).astype(np.float32)).to(self.device)
                stack_L = stack_L.reshape(self.muscle_batch_size, self.num_action, self.num_muscles)

                # [ modify ] check
                # stack_changed_m = np.vstack(batch.changed_m).astype(np.float32)
                # stack_changed_m = Tensor(stack_changed_m)

                stack_b = torch.from_numpy(np.vstack(batch.b).astype(np.float32)).to(self.device)

                if self.use_ddp:
                    activation = self.ddp_model(stack_JtA, stack_tau_des)
                else:
                    activation = self.model(stack_JtA, stack_tau_des)

                tau = torch.einsum('ijk,ik->ij', (stack_L, activation)) + stack_b

                loss_reg = activation.pow(2).mean()
                loss_target = (((tau - stack_tau_des) / 100.0).pow(2)).mean()

                loss = 0.01 * loss_reg + loss_target
                # loss = loss_target

                sum_loss += loss.item()

                self.optimizer.zero_grad()
                # loss.backward(retain_graph=True)
                loss.backward()
                self.optimizer.step()

            # print('Optimizing muscle nn : {}/{}'.format(j+1,self.num_epochs_muscle),end='\r')
        # self.loss_muscle = loss.cpu().detach().numpy().tolist()
        # print('')
        self.stats = {
            'loss_muscle': sum_loss
        }

class MarginalLearner:
    def __init__(self, num_paramstate, learning_rate=1e-3, num_epochs=10, batch_size=128, marginal_k=10.):
        self.num_paramstate = num_paramstate
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.marginal_k = marginal_k

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = MarginalNN(self.num_paramstate).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for param in self.model.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))

        self.marginal_value_avg = 1.
        self.initial_param_states = []
        self.param_sample_count = 1000

        self.stats = {}

    def get_model_weights(self) -> Dict:
        return self.model.state_dict()

    def save(self, name):
        self.model.save(name)

    def learn(self, marginal_transitions: List[MarginalTransition]):
        marginal_transitions = np.array(marginal_transitions, dtype = MarginalTransition)
        for j in range(self.num_epochs):
            np.random.shuffle(marginal_transitions)
            for i in range(len(marginal_transitions)//self.batch_size):
                transitions = marginal_transitions[i*self.batch_size:(i+1)*self.batch_size]
                batch = MarginalTransition(*zip(*transitions))

                stack_sb = np.vstack(batch.sb).astype(np.float32)
                stack_v = np.vstack(batch.v).astype(np.float32)

                v = self.model(torch.from_numpy(stack_sb).to(self.device))

                # Marginal Loss
                loss_marginal = ((v-torch.from_numpy(stack_v).to(self.device)).pow(2)).mean()
                self.marginal_loss = loss_marginal.cpu().detach().numpy().tolist()
                self.optimizer.zero_grad()
                loss_marginal.backward()

                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)
                self.optimizer.step()

                # Marginal value average
                avg_marginal = stack_v.mean()
                self.marginal_value_avg -= self.learning_rate * (self.marginal_value_avg - avg_marginal)

    def generate_initial_states(self) -> List[np.array]:
        min_v = self.env.GetParamMin()
        max_v = self.env.GetParamMax()

        # target distribution
        def target_dist(x):
            marginal_value = self.model(torch.from_numpy(x)).cpu().detach().numpy().reshape(-1)
            p = math.exp(self.marginal_k * (1. - marginal_value))
            return p

        # proposed distribution
        def proposed_dist(x, min_v, max_v):
            size = x.size
            value = np.array([np.random.uniform(min_v[i], max_v[i]) for i in range(size)])
            return value

        mcmc_sampler = mcmc.MetropolisHasting(self.num_paramstate, min_v, max_v, target_dist, proposed_dist)
        return mcmc_sampler.get_sample(self.param_sample_count)

def with_vector_env(vector_env, callable):
    envs = vector_env.get_unwrapped()
    if envs:
        return [callable(env) for env in envs]
    else:
        return callable(vector_env)

def with_worker_env(worker, callable, block=False):
    res = worker.apply.remote(
        lambda worker: with_vector_env(worker.async_env.vector_env, callable)
    )

    if block:
        return ray.get(res)
    else:
        return res

def with_multiple_worker_env(workers, callable, block=False):
    res = [worker.apply.remote(
        lambda worker: with_vector_env(worker.async_env.vector_env, callable)
    ) for worker in workers]

    if type(res[0]) == list:
        res = itertools.chain.from_iterable(res)

    if block:
        return ray.get(res)
    else:
        return res

def train(config, reporter):
    Env = MyVectorEnv if config["env_config"]["use_multi_env"] else MyEnv

    algorithm_type = config.pop("algorithm")
    num_iters = config.pop("num_iters")

    if algorithm_type == "PPO":
        trainer = PPOTrainer(config=config, env=Env)
    elif algorithm_type == "DDPPO":
        trainer = DDPPOTrainer(config=config, env=Env)
    elif algorithm_type == "IMPALA":
        trainer = ImpalaTrainer(config=config, env=Env)
    elif algorithm_type == "APPO":
        trainer = APPOTrainer(config=config, env=Env)
    elif algorithm_type == "ARS":
        trainer = ARSTrainer(config=config, env=Env)
    else:
        raise RuntimeError(f"{args.algorithm} not supported")

    local_env = trainer.workers.local_worker().env

    """
    RemoteMuscleLearner = ray.remote(MuscleLearner)
    muscle_learner = RemoteMuscleLearner.remote(
            local_env.num_action, local_env.num_muscles, local_env.num_muscle_dofs,
            run_distributed=False, use_ddp=False)

    RemoteMarginalLearner = ray.remote(MarginalLearner)
    marginal_learner = RemoteMarginalLearner.remote(
            local_env.num_paramstate)
    """

    if local_env.use_muscle:
        muscle_learner = MuscleLearner(
                local_env.num_action, local_env.num_muscles, local_env.num_muscle_dofs,
                run_distributed=False, use_ddp=False)

    if local_env.use_adaptive_sampling:
        marginal_learner = MarginalLearner(
                local_env.num_paramstate)

    mass_home = config["env_config"]["mass_home"]

    for i in range(1, num_iters + 1):
        result = trainer.train()
        reporter(**result)

        # TODO: Run both paths (muscle / adaptive_sampling) simultaneously using ray.remote()

        if local_env.use_muscle:
            remote_workers = trainer.workers.remote_workers()
            muscle_tuples = with_multiple_worker_env(remote_workers, lambda env: env.get_muscle_tuples())

            muscle_learner.learn(muscle_tuples)
            model_weights = muscle_learner.get_model_weights()

            def sync_muscle_weights(env):
                env.load_muscle_model_weights(model_weights)

            with_multiple_worker_env(remote_workers, sync_muscle_weights, block=True)

        if local_env.use_adaptive_sampling:
            remote_workers = trainer.workers.remote_workers()
            marginal_tuples = with_multiple_worker_env(remote_workers, lambda env: env.get_marginal_tuples())

            marginal_learner.learn(marginal_tuples)
            initial_param_states = marginal_learner.generate_initial_states()

            def sync_initial_param_states(env):
                env.set_initial_param_states(initial_param_states)

            with_multiple_worker_env(remote_workers, sync_initial_param_states, block=True)

        model = trainer.get_policy().model

        if i % 50 == 0:
            checkpoint = trainer.save()

            model.save(f"{mass_home}/nn/{i}.pt")
            if local_env.use_muscle:
                muscle_learner.save(f"{mass_home}/nn/{i}_muscle.pt")
            if local_env.use_adaptive_sampling:
                marginal_learner.save(f"{mass_home}/nn/{i}_marginal.pt")

from pathlib import Path

def select(cond, v1, v2):
    return v1 if cond else v2

parser = argparse.ArgumentParser()
parser.add_argument('--without_tune', help='use this in pycharm', action='store_true')
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--redis_password", type=str)
parser.add_argument("--cluster", action='store_true')
parser.add_argument("--config", type=str, default="default")
parser.add_argument("--config_file", type=str, default="python/ray_config.py")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.cluster:
        ray.init(address=os.environ["ip_head"], redis_password=args.redis_password)
        print("Nodes in the Ray cluster:")
        print(ray.nodes())
    else:
        ray.init(num_cpus=32, num_gpus=1)

    ModelCatalog.register_custom_model("my_model", SimulationNN_Ray)
    register_env("MyEnv", lambda conf: MyEnv(conf))
    if MultiEnvManager:
        register_env("MyVectorEnv", lambda conf: MyVectorEnv(conf))

    Path('nn').mkdir(exist_ok=True)

    _locals = dict()
    exec(open(args.config_file).read(), globals(), _locals)
    config = _locals["CONFIG"][args.config]

    stop_cond = {"training_iteration": config["num_iters"]}

    if args.without_tune:
       train(config, lambda *args, **kwargs: None)
    else:
        local_dir = config["env_config"]["mass_home"] + "/ray_result"
        tune.run(train,
                 name=args.config,
                 config=config,
                 local_dir=local_dir,
                 stop=stop_cond)

    ray.shutdown()
