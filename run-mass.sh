#!/bin/bash

#SBATCH -J mass
#SBATCH -p all
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH -o ./logs/mass.%j.out

pushd python
OMP_NUM_THREADS=1 /home/lasagnaphil/.conda/envs/mass/bin/python torchdist_train.py \
    --meta ../data/metadata.txt \
    --rank 0 --world-size 1 \
    --name exp-$SLURM_JOBID --run-single-node
popd
