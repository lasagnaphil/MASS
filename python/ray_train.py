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
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

from python.pymss import EnvManager
from python.Model import *

from typing import Dict

parser = argparse.ArgumentParser()
parser.add_argument('--without_tune', help='use this in pycharm', action='store_true')
parser.add_argument("--gpu", action="store_true")

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

        obs = self.env.GetStates()
        rewards = [0 if np.isnan(reward) else reward for reward in self.env.GetRewards()]
        dones = [self.env.IsEndOfEpisode(i) or np.isnan(rewards[i]) for i in range(self.num_envs)]
        infos = [{} for _ in range(self.num_envs)]

        return obs, rewards, dones, infos

    @override(VectorEnv)
    def get_unwrapped(self):
        return [None for _ in range(self.num_envs)]
        # return [MyEnv(self.config, self.observation_space, self.action_space, self.env, index) for index in range(self.num_envs)]


    def load_muscle_model_weights(self, weights):
        self.muscle_model.load_state_dict(weights)

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

from pathlib import Path

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(num_cpus=32, num_gpus=1)

    ModelCatalog.register_custom_model("my_model", SimulationNN_Ray)

    Path('../nn_ray').mkdir(exist_ok=True)

    config={
        "env": MyVectorEnv,
        "env_config": {
            "mass_home": "/home/lasagnaphil/dev/MASS",
            "meta_file": "data/metadata.txt",
            "num_envs": 32,
        },

        "num_workers": 1,
        "framework": "torch",

        # "model": {
        #     "custom_model": "my_model",
        #     "custom_model_config": {},
        #     "max_seq_len": 0    # Placeholder value needed for ray to register model
        # },


        "model": {
            "fcnet_activation": "relu", # TODO: use LeakyReLU?
            "fcnet_hiddens": [256, 256],
            "vf_share_layers": False,
        },

        "use_critic": True,
        "use_gae": True,
        "lambda": 0.99,
        "gamma": 0.99,
        "kl_coeff": 0.2,
        "rollout_fragment_length": 128,
        "train_batch_size": 4096,
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
        tune.run("PPO",
                 config=config,
                 local_dir=config["env_config"]["mass_home"] + "/ray_result",
                 checkpoint_freq=50,
                 checkpoint_at_end=True)

    ray.shutdown()
