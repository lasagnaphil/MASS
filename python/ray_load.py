import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.annotations import override
import gym
import numpy as np

class FakeVectorEnv(VectorEnv):
    def __init__(self, config):
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(config["state_dim"],))
        action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(config["action_dim"],))
        super(FakeVectorEnv, self).__init__(obs_space, action_space, 1)

    @override(VectorEnv)
    def vector_reset(self):
        pass

    @override(VectorEnv)
    def reset_at(self, index: int):
        pass

    @override(VectorEnv)
    def vector_step(self, actions):
        pass

    @override(VectorEnv)
    def get_unwrapped(self):
        pass

config={
    "env_config": {
        "mass_home": "/home/lasagnaphil/dev/MASS",
        "meta_file": "data/metadata.txt",
        "num_agents": 32,
    },

    "num_workers": 0,

    "framework": "torch",
    "model": {
        "fcnet_activation": "relu", # TODO: use LeakyReLU?
        "fcnet_hiddens": [256, 256],
        "vf_share_layers": False,
    },
}

class Evaluator:
    def __init__(self, state_dim: int, action_dim: int, checkpoint_path: str):
        ray.init()

        self.config = config.copy()
        self.config["env_config"]["state_dim"] = state_dim
        self.config["env_config"]["action_dim"] = action_dim
        self.agent = PPOTrainer(config=self.config, env=FakeVectorEnv)
        self.agent.restore(checkpoint_path)

    def get_action(self, obs):
        return self.agent.compute_action(obs)

    def shutdown(self):
        ray.shutdown()
