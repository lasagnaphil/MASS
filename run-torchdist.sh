#!/bin/bash

#SBATCH --job-name=mass_torchdist
#SBATCH --nodes=2
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task=40
#SBATCH --output ./logs/mass.%j.out

PYTHON_EXEC=/home/lasagnaphil/.conda/envs/massmpi/bin/python 

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=( $nodes )
node_count=${#nodes_array[@]}

master_addr=10.1.20.1
master_port=29500

pushd python 

for (( i=0; i<$node_count; i++ ))
do
    echo "Starting node ${nodes_array[$i]}..."

    srun --nodes=1 --ntasks=1 --cpus-per-task=40 -w ${nodes_array[$i]} $PYTHON_EXEC -m torch.distributed.launch \
        --nproc_per_node 40 \
        --nnodes $node_count \
        --node_rank $i \
        --master_addr $master_addr --master_port $master_port \
        torchdist_train.py --meta ../data/metadata.txt --name exp-$SLURM_JOBID &
done

sleep infinity

popd
