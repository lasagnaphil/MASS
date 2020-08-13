#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run_p2p_blocking(rank, world_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

def run_p2p_nonblocking(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])

def run_allreduce(rank, size):
    """ Simple point-to-point communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '10.1.20.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

    rank = dist.get_rank()
    size = dist.get_world_size()
    if type(fn) is list:
        for f in fn:
            f(rank, size)
            dist.barrier()
    else:
        fn(rank, size)

if __name__ == "__main__":
    init_process(0, 0, [run_p2p_blocking, run_p2p_nonblocking, run_allreduce], backend='mpi')
