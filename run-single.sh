#!/bin/bash

#SBATCH --job-name mass
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 40
#SBATCH --output ./logs/mass.%j.out

pushd python
# MKL_NUM_THREADS=40 OMP_NUM_THREADS=40 /home/lasagnaphil/.conda/envs/mass/bin/python torchdist_train.py \
#     --meta ../data/metadata.txt \
#     --name exp-$SLURM_JOBID --run_single_node
MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lasagnaphil/.conda/envs/mass/bin/python torchdist_train.py \
    --meta ../data/metadata.txt \
    --name exp-$SLURM_JOBID --run_single_node
popd
