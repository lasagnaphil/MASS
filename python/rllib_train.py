import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
from collections import namedtuple
from collections import deque

import numpy as np
from pymss import EnvManager
from IPython import embed
from Model import *

from typing import List, Tuple, Dict

Episode = namedtuple('Episode', ('s', 'a', 'r', 'value', 'logprob'))
MuscleTransition = namedtuple('MuscleTransition', ('JtA', 'Jtp', 'tau_des', 'L', 'b'))
Transition = namedtuple('Transition', ('s', 'a', 'logprob', 'TD', 'GAE'))
MarginalTuple = namedtuple('MarginalTuple', ('s_b', 'v'))

@ray.remote
class ReferenceLearner:
    def __init__(self, num_state, num_action):
        self.num_state = num_state
        self.num_action = num_action

        self.gamma = 0.99
        self.lb = 0.99

        self.num_epochs = 10
        self.buffer_size = 2048
        self.batch_size = 128

        self.default_learning_rate = 1E-4
        self.default_clip_ratio = 0.2
        self.learning_rate = self.default_learning_rate
        self.clip_ratio = self.default_clip_ratio

        self.w_entropy = -0.001

        self.device = torch.device("cuda")

        self.model = SimulationNN(num_state, num_action).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.num_evaluation = 0
        self.num_tuple_so_far = 0
        self.max_return = -1.0
        self.max_return_epoch = 1

        self.tic = time.time()

    def get_model_weights(self) -> Dict:
        return self.model.state_dict()

    def save(self):
        self.model.save('../nn/current.pt')
        if self.max_return_epoch == self.num_evaluation:
            self.model.save('../nn/max.pt')
        if self.num_evaluation % 100 == 0:
            self.model.save('../nn/'+str(self.num_evaluation//100)+'.pt')

    @ray.method(num_return_vals=2)
    def learn(self, episodes: List[List[Episode]]) -> Tuple[List[MarginalTuple], Dict]:
        num_episode = 0
        num_tuple = 0

        loss_actor = 0.0
        loss_critic = 0.0
        # loss_muscle = 0.0
        sum_return = 0.0

        # Convert episodes into transitions
        all_transitions: List[Transition] = []
        for data in episodes:
            size = len(data)
            if size == 0:
                continue
            states, actions, rewards, values, logprobs = zip(*data)

            values = np.concatenate((values, np.zeros(1)), axis=0)
            advantages = np.zeros(size)
            ad_t = 0

            epi_return = 0.0
            for i in reversed(range(len(data))):
                epi_return += rewards[i]
                delta = rewards[i] + values[i + 1] * self.gamma - values[i]
                ad_t = delta + self.gamma * self.lb * ad_t
                advantages[i] = ad_t
            sum_return += epi_return
            TD = values[:size] + advantages

            for i in range(size):
                all_transitions.append(Transition(states[i], actions[i], logprobs[i], TD[i], advantages[i]))

        num_episode = len(episodes)
        num_tuple = len(all_transitions)
        print('SIM : {}'.format(num_tuple))
        self.num_tuple_so_far += num_tuple

        # Optimize actor and critic
        sum_loss_actor = 0.0
        sum_loss_critic = 0.0
        for j in range(self.num_epochs):
            sum_loss_actor = 0.0
            sum_loss_critic = 0.0
            np.random.shuffle(all_transitions)
            for i in range(len(all_transitions) // self.batch_size):
                transitions = all_transitions[i * self.batch_size:(i + 1) * self.batch_size]
                batch = Transition(*zip(*transitions))

                stack_s = torch.from_numpy(np.vstack(batch.s).astype(np.float32)).to(self.device)
                stack_a = torch.from_numpy(np.vstack(batch.a).astype(np.float32)).to(self.device)
                stack_lp = torch.from_numpy(np.vstack(batch.logprob).astype(np.float32)).to(self.device)
                stack_td = torch.from_numpy(np.vstack(batch.TD).astype(np.float32)).to(self.device)
                stack_gae = torch.from_numpy(np.vstack(batch.GAE).astype(np.float32)).to(self.device)

                a_dist, v = self.model(stack_s)
                '''Critic Loss'''
                loss_critic = ((v - stack_td).pow(2)).mean()

                '''Actor Loss'''
                ratio = torch.exp(a_dist.log_prob(stack_a) - stack_lp)
                stack_gae = (stack_gae - stack_gae.mean()) / (stack_gae.std() + 1E-5)
                surrogate1 = ratio * stack_gae
                surrogate2 = torch.clamp(ratio, min=1.0 - self.clip_ratio, max=1.0 + self.clip_ratio) * stack_gae
                loss_actor = - torch.min(surrogate1, surrogate2).mean()
                '''Entropy Loss'''
                loss_entropy = - self.w_entropy * a_dist.entropy().mean()

                loss = loss_actor + loss_entropy + loss_critic

                sum_loss_actor += loss_actor.item()
                sum_loss_critic += loss_critic.item()

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)
                self.optimizer.step()
            # print('Optimizing sim nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
        # print('')

        # Calculate stats
        self.num_evaluation = self.num_evaluation + 1
        h = int((time.time() - self.tic)//3600.0)
        m = int((time.time() - self.tic)//60.0)
        s = int((time.time() - self.tic))
        m = m - h*60
        s = int((time.time() - self.tic))
        s = s - h*3600 - m*60
        if num_episode == 0:
            num_episode = 1
        if num_tuple == 0:
            num_tuple = 1
        if self.max_return < sum_return/num_episode:
            self.max_return = sum_return/num_episode
            self.max_return_epoch = self.num_evaluation

        stats = {
            'time': '{}h:{}m:{}s'.format(h,m,s),
            'num_evaluation': self.num_evaluation,
            'loss_actor': sum_loss_actor,
            'loss_critic': sum_loss_critic,
            'noise': self.model.log_std.exp().mean().item(),
            'num_transitions_so_far': self.num_tuple_so_far,
            'num_transitions': num_tuple,
            'num_episode': num_episode,
            'avg_reward_per_episode': sum_return/num_episode,
            'avg_reward_per_transition': sum_return/num_tuple,
            'avg_step_per_episode': num_tuple/num_episode,
            'max_avg_return_so_far': self.max_return,
            'max_avg_return_so_far_epoch': self.max_return_epoch
        }

        # TODO: calculate marginal tuples
        return [], stats  # (s_b, V)

@ray.remote
class MuscleLearner:
    def __init__(self, num_action, num_muscles, num_muscle_dofs):
        self.num_action = num_action
        self.num_muscles = num_muscles
        self.num_epochs_muscle = 3
        self.muscle_batch_size = 128
        self.default_learning_rate = 1E-4
        self.learning_rate = self.default_learning_rate

        self.device = torch.device("cuda")

        self.model = MuscleNN(num_muscle_dofs, self.num_action, self.num_muscles).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_model_weights(self) -> Dict:
        return self.model.state_dict()

    def save(self, save_max: bool, save_periodic: bool):
        self.model.save('../nn/current_muscle.pt')
        if save_max:
            self.model.save('../nn/max_muscle.pt')
        if save_periodic:
            self.model.save('../nn/'+str(self.num_evaluation//100)+'_muscle.pt')

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
                # Added
                stack_Jtp = torch.from_numpy(np.vstack(batch.Jtp).astype(np.float32)).to(self.device)

                stack_tau_des = torch.from_numpy(np.vstack(batch.tau_des).astype(np.float32)).to(self.device)
                stack_L = torch.from_numpy(np.vstack(batch.L).astype(np.float32)).to(self.device)
                stack_L = stack_L.reshape(self.muscle_batch_size, self.num_action, self.num_muscles)

                # [ modify ] check
                # stack_changed_m = np.vstack(batch.changed_m).astype(np.float32)
                # stack_changed_m = Tensor(stack_changed_m)

                stack_b = torch.from_numpy(np.vstack(batch.b).astype(np.float32)).to(self.device)

                activation = self.model(stack_JtA, stack_Jtp, stack_tau_des)
                tau = torch.einsum('ijk,ik->ij', (stack_L, activation)) + stack_b

                loss_reg = activation.pow(2).mean()
                loss_target = (((tau - stack_tau_des) / 100.0).pow(2)).mean()

                loss = 0.01 * loss_reg + loss_target
                # loss = loss_target

                sum_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)
                self.optimizer.step()

            # print('Optimizing muscle nn : {}/{}'.format(j+1,self.num_epochs_muscle),end='\r')
        # self.loss_muscle = loss.cpu().detach().numpy().tolist()
        # print('')
        return {
            'loss_muscle': sum_loss
        }

# TODO: Make this a dummy actor for now
@ray.remote
class BodyParamSampler:
    def __init__(self):
        pass

    def learn(self, marginal_tuples: List[MarginalTuple]) -> Dict:
        return {}

    def sample(self, num_envs: int, agents_per_env: int) -> np.array:
        return np.zeros((num_envs, agents_per_env))

@ray.remote
class VectorEnv:
    def __init__(self, meta_file: str, num_slaves: int):
        np.random.seed(seed=int(time.time()))
        self.num_slaves = num_slaves
        self.buffer_size = 10000
        self.env = EnvManager(meta_file, self.num_slaves)
        self.use_muscle = self.env.UseMuscle()
        self.num_state = self.env.GetNumState()
        self.num_action = self.env.GetNumAction()
        self.num_muscles = self.env.GetNumMuscles()
        self.num_muscle_dofs = self.env.GetNumTotalMuscleRelatedDofs()
        self.total_episodes = []

        self.counter = 0

        self.device = torch.device("cuda")

        self.ref_model = SimulationNN(self.num_state, self.num_action).to(self.device)
        self.muscle_model = MuscleNN(self.num_muscle_dofs, self.num_action, self.num_muscles).to(
            self.device)

        self.num_simulation_Hz = self.env.GetSimulationHz()
        self.num_control_Hz = self.env.GetControlHz()
        self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

    def get_params(self) -> Dict:
        return {
            'num_state': self.num_state,
            'num_action': self.num_action,
            'num_muscles': self.num_muscles,
            'num_muscle_dofs': self.num_muscle_dofs,
        }

    def reset_with_new_body(self, body_params):
        # TODO: reset envs with the specified body parameters
        for j in range(self.num_slaves):
            self.env.Reset(True, j)
        return True

    def update_model_weights(self, ref_model_weights: Dict, muscle_model_weights: Dict):
        self.ref_model.load_state_dict(ref_model_weights)
        self.muscle_model.load_state_dict(muscle_model_weights)

    def generate_tuples(self) -> Tuple[List[List[Episode]], List[MuscleTransition]]:
        total_episodes: List[List[Episode]] = []
        episodes = [[] for _ in range(self.num_slaves)]
        states = [0.0] * self.num_slaves
        actions = [0.0] * self.num_slaves
        rewards = [0.0] * self.num_slaves
        values = [0.0] * self.num_slaves
        logprobs = [0.0] * self.num_slaves
        states_next = [0.0] * self.num_slaves
        terminated = [False] * self.num_slaves

        local_step = 0
        while local_step < self.buffer_size:
            self.counter += 1
            """
            if self.counter % 10 == 0:
                print('SIM : {}'.format(self.local_step), end='\r')
            """

            states = self.env.GetStates()
            a_dist, v = self.ref_model(torch.from_numpy(states).float().to(self.device))
            actions = a_dist.sample()
            logprobs = a_dist.log_prob(actions).cpu().detach().numpy().reshape(-1)
            actions = actions.cpu().detach().numpy()
            values = v.cpu().detach().numpy().reshape(-1)

            self.env.SetActions(actions)
            if self.use_muscle:
                mt = torch.from_numpy(self.env.GetMuscleTorques()).to(self.device)
                pmt = torch.from_numpy(self.env.GetPassiveMuscleTorques()).to(self.device)
                # cm = Tensor(self.env.GetParamStates())
                for i in range(self.num_simulation_per_control // 2):
                    dt = torch.from_numpy(self.env.GetDesiredTorques()).to(self.device)

                    activations = self.muscle_model(mt, pmt, dt).cpu().detach().numpy()
                    self.env.SetActivationLevels(activations)
                    self.env.Steps(2)
            else:
                self.env.StepsAtOnce()

            for j in range(self.num_slaves):
                nan_occur = False
                terminated_state = True

                if np.any(np.isnan(states[j])) or np.any(np.isnan(actions[j])) or np.any(np.isnan(rewards[j])) \
                        or np.any(np.isnan(values[j])) or np.any(np.isnan(logprobs[j])):
                    nan_occur = True

                elif self.env.IsEndOfEpisode(j) is False:
                    terminated_state = False
                    rewards[j] = self.env.GetReward(j)
                    episodes[j].append(Episode(states[j], actions[j], rewards[j], values[j], logprobs[j]))
                    local_step += 1

                if terminated_state or nan_occur:
                    if nan_occur:
                        del episodes[j][-1]
                    total_episodes.append(episodes[j])
                    episodes[j] = []

                    self.env.Reset(True, j)

        muscle_data: List[MuscleTransition] = []
        muscle_tuples = self.env.GetMuscleTuples()
        for i in range(len(muscle_tuples)):
            muscle_data.append(MuscleTransition(
                muscle_tuples[i][0], muscle_tuples[i][1], muscle_tuples[i][2], muscle_tuples[i][3], muscle_tuples[i][4]))

        return total_episodes, muscle_data


import argparse
import os
import functools
import operator
import asyncio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model path')
    parser.add_argument('-d', '--meta', help='meta file')
    parser.add_argument('-p', '--redis_password', help='Redis password for ray')
    parser.add_argument('-n', '--nodes', help='Number of worker nodes')

    args = parser.parse_args()
    if args.meta is None:
        print('Provide meta file')
        exit()

    redis_password = args.redis_password
    num_worker_nodes = int(args.nodes)
    num_agents_per_env = 40

    ray.init(address=os.environ["ip_head"], redis_password=redis_password)

    print("Nodes in the Ray cluster:")
    print(ray.nodes())

    envs = [VectorEnv.options(num_cpus=40).remote(meta_file=args.meta, num_slaves=num_worker_nodes)
            for _ in range(num_worker_nodes)]
    env_params = ray.get(envs[0].get_params.remote())

    ref_learner = ReferenceLearner.options(num_cpus=40*num_worker_nodes) \
        .remote(env_params['num_state'], env_params['num_action'])
    muscle_learner = MuscleLearner.options(num_cpus=40*num_worker_nodes) \
        .remote(env_params['num_action'], env_params['num_muscles'], env_params['num_muscle_dofs'])
    body_param_sampler = BodyParamSampler.options(num_cpus=40).remote()

    while True:
        body_params = body_param_sampler.sample.remote(num_vector_envs, num_agents_per_env)
        for env in envs:
            ray.get(env.reset_with_new_body.remote(body_params))

        episodes, muscle_transitions = functools.reduce(
            operator.iconcat, [ray.get(env.generate_tuples.remote()) for env in envs], [])

        marginal_tuples_id, ref_stats_id = ref_learner.learn.remote(episodes)
        muscle_stats_id = muscle_learner.learn.remote(muscle_transitions)
        body_sample_stats_id = body_param_sampler.learn.remote(marginal_tuples_id)

        ref_stats = ray.get(ref_stats_id)
        muscle_stats = ray.get(muscle_stats_id)
        body_sample_stats = ray.get(body_sample_stats_id)

        # TODO: print stats
        print(ref_stats)
        print(muscle_stats)
        print(body_sample_stats)

        for env in envs:
            ray.get(env.update_model_weights.remote(ref_learner.get_model_weights.remote(),
                                                    muscle_learner.get_model_weights.remote()))

        ref_learner.save.remote()
        muscle_learner.save.remote(
            ref_stats['max_avg_return_so_far_epoch'] == ref_stats['num_evaluation'],
            ref_stats['num_evaluation'] % 100 == 0
        )

    """
    ppo = PPO(args.meta)
    nn_dir = '../nn'
    if not os.path.exists(nn_dir):
        os.makedirs(nn_dir)
    if args.model is not None:
        ppo.LoadModel(args.model)
    else:
        ppo.SaveModel()
    print('num states: {}, num actions: {}'.format(ppo.env.GetNumState(),ppo.env.GetNumAction()))
    # print('debug')
    for i in range(ppo.max_iteration-5):
        ppo.Train()
        rewards = ppo.Evaluate()
        Plot(rewards,'reward',0,False)
    """
