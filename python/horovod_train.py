import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import horovod.torch as hvd

import time
from collections import namedtuple
from collections import deque
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
from pymss import EnvManager
from Model import *

from dataclasses import dataclass

from typing import List, Tuple, Dict

MuscleTransition = namedtuple('MuscleTransition', ('JtA', 'tau_des', 'L', 'b'))
MarginalTuple = namedtuple('MarginalTuple', ('s_b', 'v'))

class Sample:
    def __init__(self, num_agents, buffer_size, state_size, action_size):
        assert buffer_size % num_agents == 0, "buffer_size must be a multiple of num_agents"

        self.num_agents = num_agents
        self.buffer_size = buffer_size
        self.num_iters = buffer_size // num_agents
        self.state_size = state_size
        self.action_size = action_size

        self.states = np.zeros((num_agents, self.num_iters, state_size), dtype=np.float32)
        self.actions = np.zeros((num_agents, self.num_iters, action_size), dtype=np.float32)
        self.rewards = np.zeros((num_agents, self.num_iters), dtype=np.float32)
        self.values = np.zeros((num_agents, self.num_iters + 1), dtype=np.float32)
        self.logprobs = np.zeros((num_agents, self.num_iters), dtype=np.float32)
        self.dones = np.zeros((num_agents, self.num_iters), dtype=np.bool)

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def flatten_buffer(buffer: np.array):
    return buffer.reshape(buffer.shape[0] * buffer.shape[1], *buffer.shape[2:])

class ReferenceLearner:
    def __init__(self, num_state, num_action, minibatch_size=128, enable_hvd=False):
        self.num_state = num_state
        self.num_action = num_action

        self.gamma = 0.99
        self.lb = 0.99

        self.num_epochs = 10
        self.minibatch_size = minibatch_size

        self.default_learning_rate = 1E-4
        self.default_clip_ratio = 0.2
        self.learning_rate = self.default_learning_rate
        self.clip_ratio = self.default_clip_ratio

        self.w_entropy = -0.001

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = SimulationNN(num_state, num_action).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if enable_hvd:
            self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=self.model.named_parameters())

        # for param in self.model.parameters():
        #     param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))

        if enable_hvd:
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        self.num_evaluation = 0
        self.num_tuple_so_far = 0
        self.max_return = -1.0
        self.max_return_epoch = 1

        self.tic = time.time()

        self.stats = {}

    def get_model_weights(self) -> Dict:
        return self.model.state_dict()

    def save(self, name):
        self.model.save(name)

    def learn(self, sample: Sample) -> List[MarginalTuple]:
        advantages = np.zeros((sample.num_agents, sample.num_iters), dtype=np.float32)

        # Convert episodes into transitions
        last_advantage = 0
        last_value = sample.values[:, -1]

        for t in reversed(range(sample.num_iters)):
            mask = 1.0 - sample.dones[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = sample.rewards[:, t] + self.gamma * last_value - sample.values[:, t]
            last_advantage = delta + self.gamma * self.lb * last_advantage
            advantages[:, t] = last_advantage
            last_value = sample.values[:, t]

        sum_return = np.sum(sample.rewards) / sample.num_agents

        td = sample.values[:, :-1] + advantages

        # Optimize actor and critic
        sum_loss_actor = 0.0
        sum_loss_critic = 0.0

        states_flat = flatten_buffer(sample.states)
        actions_flat = flatten_buffer(sample.actions)
        logprobs_flat = flatten_buffer(sample.logprobs)
        td_flat = flatten_buffer(td)
        gae_flat = flatten_buffer(advantages)

        for j in range(self.num_epochs):
            indices = torch.randperm(sample.buffer_size)

            sum_loss_actor = 0.0
            sum_loss_critic = 0.0

            for start in range(0, sample.buffer_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                s = torch.from_numpy(states_flat[mb_idx]).to(self.device)
                a = torch.from_numpy(actions_flat[mb_idx]).to(self.device)
                lp = torch.from_numpy(logprobs_flat[mb_idx]).to(self.device)
                td = torch.from_numpy(td_flat[mb_idx]).to(self.device)
                gae = torch.from_numpy(gae_flat[mb_idx]).to(self.device)

                a_dist, v = self.model(s)

                '''Critic Loss'''
                loss_critic = ((v - td).pow(2)).mean()

                '''Actor Loss'''
                ratio = torch.exp(a_dist.log_prob(a) - lp)
                gae = (gae - gae.mean()) / (gae.std() + 1E-5)
                surrogate1 = ratio * gae
                surrogate2 = torch.clamp(ratio, min=1.0 - self.clip_ratio, max=1.0 + self.clip_ratio) * gae
                loss_actor = -(torch.min(surrogate1, surrogate2).mean())
                '''Entropy Loss'''
                loss_entropy = -(self.w_entropy * a_dist.entropy().mean())

                loss = loss_actor + loss_entropy + loss_critic

                sum_loss_actor += loss_actor.item()
                sum_loss_critic += loss_critic.item()

                self.optimizer.zero_grad()
                # loss.backward(retain_graph=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
                self.optimizer.step()

        # Calculate stats
        self.num_evaluation = self.num_evaluation + 1
        self.num_tuple_so_far += sample.buffer_size

        num_episode = np.count_nonzero(sample.dones)
        num_tuple = sample.buffer_size

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

        self.stats = {
            'time': '{}h:{}m:{}s'.format(h,m,s),
            'num_evaluation': self.num_evaluation,
            'loss_actor': sum_loss_actor,
            'loss_critic': sum_loss_critic,
            'noise': self.model.get_noise(),
            'avg_reward_per_episode': sum_return/num_episode,
            'avg_reward_per_transition': sum_return/num_tuple,
            'avg_step_per_episode': num_tuple/num_episode,
            'max_avg_return_so_far': self.max_return,
            'max_avg_return_so_far_epoch': self.max_return_epoch
        }

        # TODO: calculate marginal tuples
        return [] # (s_b, V)

class MuscleLearner:
    def __init__(self, num_action, num_muscles, num_muscle_dofs, minibatch_size=128, enable_hvd=False):
        self.num_action = num_action
        self.num_muscles = num_muscles

        self.num_epochs_muscle = 3
        self.minibatch_size = minibatch_size
        self.default_learning_rate = 1E-4
        self.learning_rate = self.default_learning_rate

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = MuscleNN(num_muscle_dofs, self.num_action, self.num_muscles).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if enable_hvd:
            self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=self.model.named_parameters())

        if enable_hvd:
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

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
            for i in range(len(muscle_transitions) // self.minibatch_size):
                tuples = muscle_transitions[i * self.minibatch_size:(i + 1) * self.minibatch_size]
                batch = MuscleTransition(*zip(*tuples))

                stack_JtA = torch.from_numpy(np.vstack(batch.JtA).astype(np.float32)).to(self.device)

                stack_tau_des = torch.from_numpy(np.vstack(batch.tau_des).astype(np.float32)).to(self.device)
                stack_L = torch.from_numpy(np.vstack(batch.L).astype(np.float32)).to(self.device)
                stack_L = stack_L.reshape(self.minibatch_size, self.num_action, self.num_muscles)

                # [ modify ] check
                # stack_changed_m = np.vstack(batch.changed_m).astype(np.float32)
                # stack_changed_m = Tensor(stack_changed_m)

                stack_b = torch.from_numpy(np.vstack(batch.b).astype(np.float32)).to(self.device)

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
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
                self.optimizer.step()

            # print('Optimizing muscle nn : {}/{}'.format(j+1,self.num_epochs_muscle),end='\r')
        # self.loss_muscle = loss.cpu().detach().numpy().tolist()
        # print('')

        self.stats = {
            'loss_muscle': sum_loss
        }

# TODO: Make this a dummy actor for now
class BodyParamSampler:
    def __init__(self):
        pass

    def learn(self, marginal_tuples: List[MarginalTuple]) -> Dict:
        return {}

    def sample(self, agents_per_env: int) -> np.array:
        return np.zeros((agents_per_env,))

class VectorEnv:
    def __init__(self, meta_file: str, num_agents: int, buffer_size: int = 2048):
        np.random.seed(seed=int(time.time()))
        self.num_agents = num_agents
        self.buffer_size = buffer_size
        self.env = EnvManager(meta_file, self.num_agents)
        self.use_muscle = self.env.UseMuscle()
        self.num_state = self.env.GetNumState()
        self.num_action = self.env.GetNumAction()
        self.num_muscles = self.env.GetNumMuscles()
        self.num_muscle_dofs = self.env.GetNumTotalMuscleRelatedDofs()

        self.counter = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ref_model = SimulationNN(self.num_state, self.num_action).to(self.device)
        self.muscle_model = MuscleNN(self.num_muscle_dofs, self.num_action, self.num_muscles).to(
            self.device)

        self.num_simulation_Hz = self.env.GetSimulationHz()
        self.num_control_Hz = self.env.GetControlHz()
        self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

        self.num_transitions_so_far = 0

        self.stats = {}

    def get_params(self) -> Dict:
        return {
            'num_state': self.num_state,
            'num_action': self.num_action,
            'num_muscles': self.num_muscles,
            'num_muscle_dofs': self.num_muscle_dofs,
        }

    def reset_with_new_body(self, body_params):
        # TODO: reset envs with the specified body parameters
        for j in range(self.num_agents):
            self.env.Reset(True, j)
        return True

    def update_model_weights(self, ref_model_weights: Dict, muscle_model_weights: Dict):
        self.ref_model.load_state_dict(ref_model_weights)
        self.muscle_model.load_state_dict(muscle_model_weights)

    def generate_tuples(self) -> Tuple[Sample, List[MuscleTransition]]:
        sample = Sample(self.num_agents, self.buffer_size, self.num_state, self.num_action)

        num_episodes = 0

        for t in range(sample.num_iters):
            s = self.env.GetStates()
            a_dist, v = self.ref_model(torch.from_numpy(s).float().to(self.device))
            a = a_dist.sample()
            sample.states[:,t] = s
            sample.logprobs[:,t] = a_dist.log_prob(a).cpu().detach().numpy().reshape(-1)
            sample.actions[:,t] = a.cpu().detach().numpy()
            sample.values[:,t] = v.cpu().detach().numpy().reshape(-1)

            self.env.SetActions(sample.actions[:,t])
            if self.use_muscle:
                mt = torch.from_numpy(self.env.GetMuscleTorques()).to(self.device)
                for _ in range(self.num_simulation_per_control // 2):
                    dt = torch.from_numpy(self.env.GetDesiredTorques()).to(self.device)
                    activations = self.muscle_model(mt, dt).cpu().detach().numpy()
                    self.env.SetActivationLevels(activations)
                    self.env.Steps(2)
            else:
                self.env.StepsAtOnce()

            for j in range(sample.num_agents):
                terminated_state = self.env.IsEndOfEpisode(j)
                nan_occur = np.any(np.isnan(sample.states[j,t])) or np.any(np.isnan(sample.actions[j,t])) or \
                            np.isnan(sample.values[j,t]) or np.isnan(sample.logprobs[j,t])

                sample.rewards[j,t] = self.env.GetReward(j)
                sample.dones[j,t] = terminated_state or nan_occur

                if terminated_state or nan_occur:
                    num_episodes += 1
                    self.env.Reset(True, j)

        # Gather the final values (Needed for GAE calculation)
        s = self.env.GetStates()
        _, v = self.ref_model(torch.from_numpy(s).float().to(self.device))
        sample.values[:,-1] = v.cpu().detach().numpy().reshape(-1)

        muscle_tuples = self.env.GetMuscleTuples()
        muscle_transitions = [MuscleTransition(*muscle_tuple) for muscle_tuple in muscle_tuples]

        num_transitions = sample.buffer_size
        self.num_transitions_so_far += num_transitions

        self.stats = {
            'num_episodes': num_episodes,
            'num_transitions': num_transitions,
            'num_transitions_so_far': self.num_transitions_so_far
        }
        return sample, muscle_transitions

def plot_rewards(filename, y, title, num_fig=1, ylim=True):
    temp_y = np.zeros(y.shape)
    if y.shape[0]>5:
        temp_y[0] = y[0]
        temp_y[1] = 0.5*(y[0] + y[1])
        temp_y[2] = 0.3333*(y[0] + y[1] + y[2])
        temp_y[3] = 0.25*(y[0] + y[1] + y[2] + y[3])
        for i in range(4,y.shape[0]):
            temp_y[i] = np.sum(y[i-4:i+1])*0.2

    plt.figure(num_fig)
    plt.clf()
    plt.title(title)
    plt.plot(y,'b')
    
    plt.plot(temp_y,'r')

    plt.show()
    if ylim:
        plt.ylim([0,1])

    plt.pause(0.001)

    plt.savefig(filename)

import argparse
import os
import functools
import operator
import asyncio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--enable_hvd', help='Enable horovod', action='store_true')
    parser.set_defaults(enable_hvd=False)
    parser.add_argument('-m', '--model', help='model path')
    parser.add_argument('-d', '--meta', help='meta file')
    # parser.add_argument('-r', '--rank', help='node rank')
    # parser.add_argument('-w', '--world-size', help='world size')
    parser.add_argument('-n', '--name', help='exp name', default='default')
    parser.add_argument('-b', '--batch_size', help='batch size', type=int)
    parser.add_argument('-a', '--num_agents', help='num agents', type=int)

    args = parser.parse_args()
    if args.meta is None:
        print('Provide meta file')
        exit()

    # rank = int(args.rank)
    # world_size = int(args.world_size)

    # os.environ['MASTER_ADDR'] = '10.1.20.1'
    # os.environ['MASTER_PORT'] = '29500'

    if args.enable_hvd:
        hvd.init()
        rank = hvd.rank()
        world_size = hvd.size()
        print(f'Initialized node with rank {rank}!', flush=True)
    else:
        rank = 0
        world_size = 1

    num_agents = args.num_agents
    buffer_size = args.batch_size

    env = VectorEnv(meta_file=args.meta, num_agents=num_agents, buffer_size=buffer_size)
    env_params = env.get_params()

    ref_learner = ReferenceLearner(env_params['num_state'], env_params['num_action'], enable_hvd=args.enable_hvd)
    muscle_learner = MuscleLearner(env_params['num_action'], env_params['num_muscles'], env_params['num_muscle_dofs'],
                                   enable_hvd=args.enable_hvd)
    body_param_sampler = BodyParamSampler()

    Path('../plot').mkdir(exist_ok=True)
    Path('../nn').mkdir(exist_ok=True)
    Path(f'../nn/{args.name}').mkdir(exist_ok=True)

    rewards = []

    plt.ion()

    while True:
        body_params = body_param_sampler.sample(num_agents)
        env.reset_with_new_body(body_params)

        start = time.time()
        sample, muscle_transitions = env.generate_tuples()
        end = time.time()
        if rank == 0: print(f"Generating Tuples: {end - start}s")

        start = time.time()
        marginal_tuples = ref_learner.learn(sample)
        end = time.time()
        if rank == 0: print(f"Ref learner: {end - start}s")

        start = time.time()
        muscle_learner.learn(muscle_transitions)
        end = time.time()
        if rank == 0: print(f"Muscle learner: {end - start}s")

        body_param_sampler.learn(marginal_tuples)

        if rank == 0:
            print('# {} === {} ==='.format(ref_learner.stats['num_evaluation'], ref_learner.stats['time']))
            print('||Loss Actor               : {:.4f}'.format(ref_learner.stats['loss_actor']))
            print('||Loss Critic              : {:.4f}'.format(ref_learner.stats['loss_critic']))
            print('||Loss Muscle              : {:.4f}'.format(muscle_learner.stats['loss_muscle']))
            print('||Noise                    : {:.3f}'.format(ref_learner.stats['noise']))
            print('||Num Transition So far    : {}'    .format(env.stats['num_transitions_so_far']))
            print('||Num Transition           : {}'    .format(env.stats['num_transitions']))
            print('||Num Episode              : {}'    .format(env.stats['num_episodes']))
            print('||Avg Reward per episode   : {:.3f}'.format(ref_learner.stats['avg_reward_per_episode']))
            print('||Avg Reward per transition: {:.3f}'.format(ref_learner.stats['avg_reward_per_transition']))
            print('||Avg Step per episode     : {:.1f}'.format(ref_learner.stats['avg_step_per_episode']))
            print('||Max Avg Retun So far     : {:.3f} at #{}'.format(
                ref_learner.stats['max_avg_return_so_far'], ref_learner.stats['max_avg_return_so_far_epoch']))

            rewards.append(ref_learner.stats['avg_reward_per_episode'])

            ref_learner.save(f'../nn/{args.name}/current.pt')
            muscle_learner.save(f'../nn/{args.name}/current_muscle.pt')

            if ref_learner.max_return_epoch == ref_learner.num_evaluation:
                ref_learner.save(f'../nn/{args.name}/max.pt')
                muscle_learner.save(f'../nn/{args.name}/max_muscle.pt')

            if ref_learner.num_evaluation % 100 == 0:
                ref_learner.save(f'../nn/{args.name}/{ref_learner.num_evaluation//100}.pt')
                muscle_learner.save(f'../nn/{args.name}/{ref_learner.num_evaluation//100}_muscle.pt')

            plot_rewards(f'../plot/{args.name}.png', np.array(rewards), 'reward', 0, False)

            # Flushing is needed for SLURM log output!
            print('', flush=True)

        env.update_model_weights(ref_learner.get_model_weights(), muscle_learner.get_model_weights())
        
