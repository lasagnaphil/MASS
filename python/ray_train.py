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
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

from python.pymss import EnvManager
from python.Model import *

from typing import Dict, List, Callable
from collections import namedtuple

parser = argparse.ArgumentParser()
parser.add_argument('--without_tune', help='use this in pycharm', action='store_true')
parser.add_argument("--gpu", action="store_true")

MuscleTransition = namedtuple('MuscleTransition', ('JtA', 'tau_des', 'L', 'b'))

class MyVectorEnv(VectorEnv):
    def __init__(self, config):
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(130,))
        action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(50,))
        super(MyVectorEnv, self).__init__(obs_space, action_space, config["num_envs"])

        self.config = config

        self.meta_file = config["mass_home"] + "/" + config["meta_file"]

        self.env = EnvManager(self.meta_file, self.num_envs)

        self.use_muscle = self.env.UseMuscle()
        self.num_state = self.env.GetNumState()
        self.num_action = self.env.GetNumAction()
        self.num_muscles = self.env.GetNumMuscles()
        self.num_muscle_dofs = self.env.GetNumTotalMuscleRelatedDofs()

        self.counter = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.muscle_model = MuscleNN(self.num_muscle_dofs, self.num_action, self.num_muscles).to(
            self.device)
        self.muscle_tuples = []

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
        super(SimulationNN_Ray, self).__init__(num_states, num_actions)
        self._last_value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        x = obs.reshape(obs.shape[0], -1)
        action_dist, self._last_value = super(SimulationNN_Ray, self).forward(x)
        return torch.cat([action_dist.loc, action_dist.scale], dim=1), state

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

def with_worker_env(worker, callable):
    return ray.get(worker.apply.remote(
        lambda worker: callable(worker.async_env.vector_env)
    ))

def train_ppo(config, reporter):
    trainer = PPOTrainer(config=config, env=MyVectorEnv)

    local_env = trainer.workers.local_worker().env

    muscle_learner = MuscleLearner(local_env.num_action, local_env.num_muscles, local_env.num_muscle_dofs,
                                   run_distributed=False, use_ddp=False)

    mass_home = config["env_config"]["mass_home"]

    for i in range(0, 10000):
        result = trainer.train()
        reporter(**result)

        remote_worker = trainer.workers.remote_workers()[0]
        muscle_tuples = with_worker_env(remote_worker, lambda env: env.get_muscle_tuples())

        # muscle_tuples = ray.get(
        #     [w.apply.remote(worker_get_muscle_tuple) for w in trainer.workers.remote_workers()])

        muscle_learner.learn(muscle_tuples)
        with_worker_env(remote_worker, lambda env: env.load_muscle_model_weights(
            muscle_learner.get_model_weights()))

        if i % 50 == 0:
            checkpoint = trainer.save()

            model = trainer.get_policy().model

            model.save(f"{mass_home}/nn/{i}.pt")
            muscle_learner.save(f"{mass_home}/nn/{i}_muscle.pt")


from pathlib import Path

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(num_cpus=16, num_gpus=1)

    ModelCatalog.register_custom_model("my_model", SimulationNN_Ray)

    Path('../nn_ray').mkdir(exist_ok=True)

    config={
        "env": MyVectorEnv,
        "env_config": {
            "mass_home": "/home/lasagnaphil/dev/MASS-ray",
            "meta_file": "data/metadata.txt",
            "num_envs": 16,
        },

        "num_workers": 1,
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

        "use_critic": True,
        "use_gae": True,
        "lambda": 0.99,
        "gamma": 0.99,
        "kl_coeff": 0.2,
        "rollout_fragment_length": 128,
        "train_batch_size": 2048,
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

    if args.without_tune:
        train_ppo(config, lambda *args, **kwargs: None)
    else:
        tune.run(train_ppo,
                 config=config,
                 local_dir=config["env_config"]["mass_home"] + "/ray_result",
                 checkpoint_freq=50,
                 checkpoint_at_end=True)

    ray.shutdown()
