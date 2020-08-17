#!/bin/bash

#SBATCH --job-name=mass_mpi
#SBATCH --nodes=2
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task=40
#SBATCH --output ./logs/mass.%j.out

PYTHON_EXEC=/home/lasagnaphil/.conda/envs/mass-mpi/bin/python 

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=( $nodes )
node_count=${#nodes_array[@]}

pushd python 

export MASTER_ADDR=${nodes_array[0]}
export MASTER_PORT=29500

prun $PYTHON_EXEC torchdist_test.py

popd
