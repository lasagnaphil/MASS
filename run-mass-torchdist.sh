#!/bin/bash

#SBATCH --job-name=mass_torchdist
#SBATCH --nodes=8
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task=40
#SBATCH --output ./logs/mass.%j.out

PYTHON_EXEC=/home/lasagnaphil/.conda/envs/mass/bin/python 

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=( $nodes )
worker_num=${#nodes_array[@]}

pushd python 

for (( i=0; i<$worker_num; i++ ))
do
    echo "Starting node ${nodes_array[$i]}..."

    srun --nodes=1 --ntasks=1 --cpus-per-task=40 -w ${nodes_array[$i]} $PYTHON_EXEC torchdist_train.py \
        --meta ../data/metadata.txt \
        --rank $i --world-size $worker_num \
        --name exp-$SLURM_JOBID &
done

sleep infinity

popd
