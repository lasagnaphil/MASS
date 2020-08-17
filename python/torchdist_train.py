import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import time
from collections import namedtuple
from collections import deque
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
from pymss import EnvManager
from Model import *

from typing import List, Tuple, Dict

Episode = namedtuple('Episode', ('s', 'a', 'r', 'value', 'logprob'))
MuscleTransition = namedtuple('MuscleTransition', ('JtA', 'tau_des', 'L', 'b'))
Transition = namedtuple('Transition', ('s', 'a', 'logprob', 'TD', 'GAE'))
MarginalTuple = namedtuple('MarginalTuple', ('s_b', 'v'))

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

class ReferenceLearner:
    def __init__(self, num_state, num_action, run_distributed=False, use_ddp=False, rank=0):
        self.num_state = num_state
        self.num_action = num_action
        self.run_distributed = run_distributed
        self.use_ddp = use_ddp
        self.rank = rank

        self.gamma = 0.99
        self.lb = 0.99

        self.num_epochs = 10
        self.batch_size = 128

        self.default_learning_rate = 1E-4
        self.default_clip_ratio = 0.2
        self.learning_rate = self.default_learning_rate
        self.clip_ratio = self.default_clip_ratio

        self.w_entropy = -0.001

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = SimulationNN(num_state, num_action).to(self.device)
        if self.use_ddp:
            self.ddp_model = DDP(self.model)
            self.model = self.ddp_model.module
            parameters = self.ddp_model.parameters()
        else:
            parameters = self.model.parameters()

        self.optimizer = optim.Adam(parameters, lr=self.learning_rate)
        for param in parameters:
            param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))

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

    def learn(self, episodes: List[List[Episode]]) -> Tuple[List[MarginalTuple]]:
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
        # print('SIM : {}'.format(num_tuple), flush=True)
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

                if self.use_ddp:
                    a_dist, v = self.ddp_model(stack_s)
                else:
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
                # loss.backward(retain_graph=True)
                loss.backward()
                if self.run_distributed and not self.use_ddp:
                    average_gradients(self.model)
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

        self.stats = {
            'time': '{}h:{}m:{}s'.format(h,m,s),
            'num_evaluation': self.num_evaluation,
            'loss_actor': sum_loss_actor,
            'loss_critic': sum_loss_critic,
            'noise': self.model.get_noise(),
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
        return [] # (s_b, V)

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
                if self.run_distributed and not self.use_ddp:
                    average_gradients(self.model)
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
    def __init__(self, meta_file: str, num_slaves: int, buffer_size: int = 2048):
        np.random.seed(seed=int(time.time()))
        self.num_slaves = num_slaves
        self.buffer_size = buffer_size
        self.env = EnvManager(meta_file, self.num_slaves)
        self.use_muscle = self.env.UseMuscle()
        self.num_state = self.env.GetNumState()
        self.num_action = self.env.GetNumAction()
        self.num_muscles = self.env.GetNumMuscles()
        self.num_muscle_dofs = self.env.GetNumTotalMuscleRelatedDofs()
        self.total_episodes = []

        self.counter = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ref_model = SimulationNN(self.num_state, self.num_action).to(self.device)
        self.muscle_model = MuscleNN(self.num_muscle_dofs, self.num_action, self.num_muscles).to(
            self.device)

        self.num_simulation_Hz = self.env.GetSimulationHz()
        self.num_control_Hz = self.env.GetControlHz()
        self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

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
            if self.counter % 10 == 0:
                pass
                # print('SIM : {}'.format(local_step))

            states = self.env.GetStates()
            a_dist, v = self.ref_model(torch.from_numpy(states).float().to(self.device))
            actions = a_dist.sample()
            logprobs = a_dist.log_prob(actions).cpu().detach().numpy().reshape(-1)
            actions = actions.cpu().detach().numpy()
            values = v.cpu().detach().numpy().reshape(-1)

            self.env.SetActions(actions)
            if self.use_muscle:
                mt = torch.from_numpy(self.env.GetMuscleTorques()).to(self.device)
                for i in range(self.num_simulation_per_control // 2):
                    dt = torch.from_numpy(self.env.GetDesiredTorques()).to(self.device)
                    activations = self.muscle_model(mt, dt).cpu().detach().numpy()
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
                muscle_tuples[i][0], muscle_tuples[i][1], muscle_tuples[i][2], muscle_tuples[i][3]))

        return total_episodes, muscle_data

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

    plt.savefig(filename)

import argparse
import os
import functools
import operator
import asyncio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('-m', '--model', help='model path')
    parser.add_argument('-d', '--meta', help='meta file')
    # parser.add_argument('-r', '--rank', help='node rank')
    # parser.add_argument('-w', '--world-size', help='world size')
    parser.add_argument('-n', '--name', help='exp name', default='default')
    parser.add_argument('--run_single_node', help='run single node (test without mpi)', action='store_true')
    parser.set_defaults(run_single_node=False)
    parser.add_argument('--use_mpi', help='Use MPI backend for torch', action='store_true')
    parser.set_defaults(use_mpi=False)
    parser.add_argument('--use_ddp', help='Use DDP for torch', action='store_true')
    parser.set_defaults(use_ddp=False)

    args = parser.parse_args()
    if args.meta is None:
        print('Provide meta file')
        exit()

    # rank = int(args.rank)
    # world_size = int(args.world_size)

    # os.environ['MASTER_ADDR'] = '10.1.20.1'
    # os.environ['MASTER_PORT'] = '29500'

    if not args.run_single_node:
        if args.use_mpi:
            # mpi backend
            dist_backend = 'mpi'
        else:
            # gloo / nccl backend
            if torch.cuda.is_available():
                dist_backend = 'nccl'
            else:
                dist_backend = 'gloo'
        dist.init_process_group(dist_backend)
        rank = torch.distributed.get_rank()
        print(f'Initialized node with rank {rank}!', flush=True)

    num_agents_per_env = 40
    if args.run_single_node:
        buffer_size = 2048
    else:
        buffer_size = 2048 // torch.distributed.get_world_size()

    env = VectorEnv(meta_file=args.meta, num_slaves=num_agents_per_env, buffer_size=buffer_size)
    env_params = env.get_params()

    use_distributed = not args.run_single_node
    ref_learner = ReferenceLearner(env_params['num_state'], env_params['num_action'], use_distributed, args.use_ddp)
    muscle_learner = MuscleLearner(env_params['num_action'], env_params['num_muscles'], env_params['num_muscle_dofs'], use_distributed, args.use_ddp)
    body_param_sampler = BodyParamSampler()

    Path('../plot').mkdir(exist_ok=True)
    Path('../nn').mkdir(exist_ok=True)
    Path(f'../nn/{args.name}').mkdir(exist_ok=True)

    rewards = []

    while True:
        body_params = body_param_sampler.sample(num_agents_per_env)
        env.reset_with_new_body(body_params)

        start = time.time()
        episodes, muscle_transitions = env.generate_tuples()
        end = time.time()
        if not use_distributed or rank == 0: print(f"Generating Tuples: {end - start}s")

        start = time.time()
        marginal_tuples = ref_learner.learn(episodes)
        end = time.time()
        if not use_distributed or rank == 0: print(f"Ref learner: {end - start}s")

        start = time.time()
        muscle_learner.learn(muscle_transitions)
        end = time.time()
        if not use_distributed or rank == 0: print(f"Muscle learner: {end - start}s")

        body_param_sampler.learn(marginal_tuples)

        if not use_distributed or rank == 0:
            print('# {} === {} ==='.format(ref_learner.stats['num_evaluation'], ref_learner.stats['time']))
            print('||Loss Actor               : {:.4f}'.format(ref_learner.stats['loss_actor']))
            print('||Loss Critic              : {:.4f}'.format(ref_learner.stats['loss_critic']))
            print('||Loss Muscle              : {:.4f}'.format(muscle_learner.stats['loss_muscle']))
            print('||Noise                    : {:.3f}'.format(ref_learner.stats['noise']))		
            print('||Num Transition So far    : {}'    .format(ref_learner.stats['num_transitions_so_far']))
            print('||Num Transition           : {}'    .format(ref_learner.stats['num_transitions']))
            print('||Num Episode              : {}'    .format(ref_learner.stats['num_episode']))
            print('||Avg Return per episode   : {:.3f}'.format(ref_learner.stats['avg_reward_per_episode']))
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
        
