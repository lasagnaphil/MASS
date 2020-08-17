#!/bin/bash

#SBATCH --job-name mass_mpi
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 40
#SBATCH --output ./logs/mass.%j.out

PYTHON_EXEC=/home/lasagnaphil/.conda/envs/mass-mpi/bin/python 

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=( $nodes )
node_count=${#nodes_array[@]}

pushd python 

srun -n $node_count $PYTHON_EXEC torchdist_train.py \
    --meta ../data/metadata.txt --name exp-$SLURM_JOBID

popd
