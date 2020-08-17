#!/bin/bash

#SBATCH --job-name mass_mpi
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 40
#SBATCH --output ./logs/mass.%j.out

HOROVODRUN_EXEC=/home/lasagnaphil/.conda/envs/mass-mpi/bin/horovodrun
PYTHON_EXEC=/home/lasagnaphil/.conda/envs/mass-mpi/bin/python 

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=( $nodes )
node_count=${#nodes_array[@]}

pushd python 

mpirun -np $node_count \
    -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x HOROVOD_MPI_THREADS_DISABLE=1 -x PATH \
    -mca pml ob1 \
    $PYTHON_EXEC horovod_train.py --meta ../data/metadata.txt --name exp-$SLURM_JOBID

popd
