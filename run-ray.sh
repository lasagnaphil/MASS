#!/bin/bash

#SBATCH --job-name mass_ray
#SBATCH --nodes 2
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 40
#SBATCH --output ./logs/mass.%j.out

PYTHON_EXEC=/home/lasagnaphil/.conda/envs/mass/bin/python 
RAY_EXEC=/home/lasagnaphil/.conda/envs/mass/bin/ray

worker_num=1 # Must be one less that the total number of nodes

# module load Langs/Python/3.6.4 # This will vary depending on your environment
# source venv/bin/activate

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py

# Starting the head
srun --nodes=1 --ntasks=1 -w $node1 $RAY_EXEC start --block --head --redis-port=6379 --redis-password=$redis_password \
    --include-webui=true --webui-host=0.0.0.0 &
    
sleep 5
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 $RAY_EXEC start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
done

sleep 5

$PYTHON_EXEC -u python/ray_train.py --cluster --redis_password $redis_password --algorithm=ppo

