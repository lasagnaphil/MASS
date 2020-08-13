#!/bin/bash

#SBATCH -J mass
#SBATCH -p all
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH -o ./logs/mass.%j.out

pushd python
# MKL_NUM_THREADS=40 OMP_NUM_THREADS=40 /home/lasagnaphil/.conda/envs/mass/bin/python torchdist_train.py \
#     --meta ../data/metadata.txt \
#     --name exp-$SLURM_JOBID --run_single_node
/home/lasagnaphil/.conda/envs/mass/bin/python torchdist_train.py \
    --meta ../data/metadata.txt \
    --name exp-$SLURM_JOBID --run_single_node
popd
