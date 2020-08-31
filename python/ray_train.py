import argparse
import gym
import numpy as np

import ray
from ray import tune
from ray.tune.logger import pretty_print
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.ppo import PPOTorchPolicy, PPOTrainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ars import ARSTorchPolicy, ARSTrainer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

from pymss import EnvManager, SingleEnvManager
from Model import *

from typing import Dict, List, Callable
from collections import namedtuple

import os

parser = argparse.ArgumentParser()
parser.add_argument('--without_tune', help='use this in pycharm', action='store_true')
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--redis_password", type=str)
parser.add_argument("--cluster", action='store_true')
parser.add_argument("--algorithm", type=str, default="ppo")

MuscleTransition = namedtuple('MuscleTransition', ('JtA', 'tau_des', 'L', 'b'))

class MyEnv(gym.Env):
    def __init__(self, config):
        self.meta_file = config["mass_home"] + "/" + config["meta_file"]
        self.env = SingleEnvManager(self.meta_file)

        self.use_muscle = self.env.UseMuscle()
        self.num_state = self.env.GetNumState()
        self.num_action = self.env.GetNumAction()
        self.num_muscles = self.env.GetNumMuscles()
        self.num_muscle_dofs = self.env.GetNumTotalMuscleRelatedDofs()

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_state,))
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_action,))

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.muscle_model = MuscleNN(self.num_muscle_dofs, self.num_action, self.num_muscles).to(
            self.device)

        self.num_simulation_Hz = self.env.GetSimulationHz()
        self.num_control_Hz = self.env.GetControlHz()
        self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

        self.counter = 0
        self.num_transitions_so_far = 0
        self.stats = {}

    def reset(self):
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

        return obs, reward, done, info

    def load_muscle_model_weights(self, weights):
        self.muscle_model.load_state_dict(weights)

    def get_muscle_tuples(self):
        muscle_tuples = self.env.GetMuscleTuples()
        return [MuscleTransition(t[0], t[1], t[2], t[3]) for t in muscle_tuples]

class MyVectorEnv(VectorEnv):
    def __init__(self, config):
        self.meta_file = config["mass_home"] + "/" + config["meta_file"]
        self.env = EnvManager(self.meta_file, config["num_envs"])

        self.use_muscle = self.env.UseMuscle()
        self.num_state = self.env.GetNumState()
        self.num_action = self.env.GetNumAction()
        self.num_muscles = self.env.GetNumMuscles()
        self.num_muscle_dofs = self.env.GetNumTotalMuscleRelatedDofs()

        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_state,))
        action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_action,))
        super(MyVectorEnv, self).__init__(obs_space, action_space, config["num_envs"])

        self.config = config

        self.counter = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.muscle_model = MuscleNN(self.num_muscle_dofs, self.num_action, self.num_muscles).to(
            self.device)

        self.num_simulation_Hz = self.env.GetSimulationHz()
        self.num_control_Hz = self.env.GetControlHz()
        self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

        self.num_transitions_so_far = 0

        self.stats = {}

    @override(VectorEnv)
    def vector_reset(self):
        for j in range(self.num_envs):
            self.env.Reset(True, j)
        return self.env.GetStates()

    @override(VectorEnv)
    def reset_at(self, index: int):
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

        return obs, rewards, dones, infos

    @override(VectorEnv)
    def get_unwrapped(self):
        return None
        # return [None for _ in range(self.num_envs)]
        # return [MyEnv(self.config, self.observation_space, self.action_space, self.env, index) for index in range(self.num_envs)]

    def load_muscle_model_weights(self, weights):
        self.muscle_model.load_state_dict(weights)

    def get_muscle_tuples(self):
        muscle_tuples = self.env.GetMuscleTuples()
        return [MuscleTransition(t[0], t[1], t[2], t[3]) for t in muscle_tuples]

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

def with_worker_env(worker, callable, block=False):
    res = worker.apply.remote(
        lambda worker: callable(worker.async_env.vector_env)
    )
    if block:
        return ray.get(res)
    else:
        return res

def with_multiple_worker_env(workers, callable, block=False):
    res = [worker.apply.remote(
        lambda worker: callable(worker.async_env.vector_env)
    ) for worker in workers]
    if block:
        return ray.get(res)
    else:
        return res

def train_ppo(config, reporter):
    Env = MyVectorEnv if config["env_config"]["use_multi_env"] else MyEnv

    if args.algorithm == "ppo":
        trainer = PPOTrainer(config=config, env=Env)
    elif args.algorithm == "impala":
        trainer = ImpalaTrainer(config=config, env=Env)
    elif args.algorithm == "ars":
        trainer = ARSTrainer(config=config, env=Env)
    else:
        raise RuntimeError(f"{args.algorithm} not supported")

    local_env = trainer.workers.local_worker().env

    RemoteMuscleLearner = ray.remote(MuscleLearner)
    muscle_learner = RemoteMuscleLearner.remote(
            local_env.num_action, local_env.num_muscles, local_env.num_muscle_dofs,
            run_distributed=False, use_ddp=False)

    mass_home = config["env_config"]["mass_home"]

    for i in range(1, 10001):
        result = trainer.train()
        reporter(**result)

        if local_env.use_muscle:
            remote_workers = trainer.workers.remote_workers()
            muscle_tuples = with_multiple_worker_env(remote_workers, lambda env: env.get_muscle_tuples())

            ray.get(muscle_learner.learn.remote(muscle_tuples))

            def sync_muscle_weights(env):
                model_weights = ray.get(muscle_learner.get_model_weights.remote())
                env.load_muscle_model_weights(model_weights)

            with_worker_env(remote_workers, sync_muscle_weights, block=True)

        if i % 50 == 0:
            checkpoint = trainer.save()

            model = trainer.get_policy().model
            model.save(f"{mass_home}/nn_ray/{i}.pt")

            if local_env.use_muscle:
                ray.get(muscle_learner.save.remote(f"{mass_home}/nn_ray/{i}_muscle.pt"))

from pathlib import Path

def select(cond, v1, v2):
    return v1 if cond else v2

if __name__ == "__main__":
    args = parser.parse_args()
    if args.cluster:
        ray.init(address=os.environ["ip_head"], redis_password=args.redis_password)
        print("Nodes in the Ray cluster:")
        print(ray.nodes())
    else:
        ray.init(num_cpus=32, num_gpus=1)

    ModelCatalog.register_custom_model("my_model", SimulationNN_Ray)

    Path('nn_ray').mkdir(exist_ok=True)

    if args.cluster:
        env_config = {
            "mass_home": os.environ["PWD"],
            "meta_file": "data/metadata_nomuscle.txt",
            "use_multi_env": False,
            # "num_envs": 16,
        }
    else:
        env_config = {
            "mass_home": "/home/lasagnaphil/dev/MASS-ray",
            "meta_file": "data/metadata_nomuscle.txt",
            "use_multi_env": False
            # "num_envs": 16,
        }

    config={
        "env": MyEnv,
        "env_config": env_config,

        "num_workers": 16,
        "framework": "torch",
        # "num_cpus_per_worker": env_config["num_envs"],

        "model": {
            "custom_model": "my_model",
            "custom_model_config": {},
            "max_seq_len": 0    # Placeholder value needed for ray to register model
        },

        # "model": {
        #     "fcnet_activation": "relu", # TODO: use LeakyReLU?
        #     "fcnet_hiddens": [256, 256],
        #     "vf_share_layers": False,
        # },

        "use_critic": True,
        "use_gae": True,
        "lambda": 0.99,
        "gamma": 0.99,
        "kl_coeff": 0.2,
        "rollout_fragment_length": 128,
        "train_batch_size": 16 * 128,
        "sgd_minibatch_size": 128,
        "shuffle_sequences": True,
        "num_sgd_iter": 10,
        "lr": 1e-4,
        "lr_schedule": None,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": 0.0,
        "entropy_coeff_schedule": None,
        "clip_param": 0.2,
        "vf_clip_param": 10.0,
        "grad_clip": None,
        "kl_target": 0.01,
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "simple_optimizer": False,
    }

    impala_config = {
        "env": MyEnv,
        "env_config": env_config,

        "num_workers": 16,
        "framework": "torch",

        "model": {
            "custom_model": "my_model",
            "custom_model_config": {},
            "max_seq_len": 0    # Placeholder value needed for ray to register model
        },

        # "model": {
        #     "fcnet_activation": "relu", # TODO: use LeakyReLU?
        #     "fcnet_hiddens": [256, 256],
        #     "vf_share_layers": False,
        # },

        # "use_critic": True,
        # "use_gae": True,
        # "lambda": 0.99,
        "gamma": 0.99,
        # "clip_param": 0.2,
        # "kl_coeff": 0.2,
        "rollout_fragment_length": 128,
        "train_batch_size": 512,
        "min_iter_time_s": 10,
        "num_data_loader_buffers": 1,
        "minibatch_buffer_size": 1,
        "num_sgd_iter": 10,
        "replay_proportion": 0.0,
        "replay_buffer_num_slots": 100,
        "learner_queue_size": 16,
        "learner_queue_timeout": 300,
        "max_sample_requests_in_flight_per_worker": 2,
        "broadcast_interval": 1,
        "grad_clip": 40.0,
        "opt_type": "adam",
        "lr": 0.0001,
        "lr_schedule": None,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": 0.0,
        "entropy_coeff_schedule": None,
    }

    ars_config = {
        "env": MyEnv,
        "env_config": env_config,

        "framework": "torch",

        "model": {
            "custom_model": "my_model",
            "custom_model_config": {},
            "max_seq_len": 0
        },

        "action_noise_std": 0.0,
        "noise_stdev": tune.grid_search([0.01, 0.0075, 0.005]),  # std deviation of parameter noise
        "num_rollouts": tune.grid_search([180, 270, 360, 450]),  # number of perturbs to try
        "rollouts_used": tune.grid_search([90, 180, 270]),  # number of perturbs to keep in gradient estimate
        "num_workers": 30,
        "sgd_stepsize": tune.grid_search([0.01, 0.02, 0.025]),  # sgd step-size
        "observation_filter": "MeanStdFilter",
        "noise_size": 250000000,
        "eval_prob": 0.03,  # probability of evaluating the parameter rewards
        "report_length": 10,  # how many of the last rewards we average over
        "offset": 0,
    }

    stop_cond = {"training_iteration": 100}

    if args.without_tune:
        train_ppo(config, lambda *args, **kwargs: None)
    else:
        if args.algorithm == "ppo":
            tune.run("PPO",
                     config=config,
                     local_dir=config["env_config"]["mass_home"] + "/ray_result",
                     stop=stop_cond)
        elif args.algorithm == "impala":
            tune.run("IMPALA",
                     config=impala_config,
                     local_dir=config["env_config"]["mass_home"] + "/ray_result",
                     stop=stop_cond)
        elif args.algorithm == "ars":
            tune.run("ARS",
                     config=ars_config,
                     local_dir=config["env_config"]["mass_home"] + "/ray_result",
                     stop=stop_cond)
        else:
            raise RuntimeError(f"{args.algorithm} not supported")


    ray.shutdown()
