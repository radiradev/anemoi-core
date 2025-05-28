#!/bin/bash

#SBATCH -A DestE_340_25
#SBATCH -p boost_usr_prod
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0G
#SBATCH --time=24:00:00
#SBATCH --switches=1
#SBATCH --output=outputs/single23.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export CUDA_LAUNCH_BLOCKING=1
#export HYDRA_FULL_ERROR=1
#export NCCL_DEBUG=TRACE
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# generic settings

export ANEMOI_BASE_SEED=16014075
GITDIR=/leonardo/home/userexternal/epinning/aifs/anemoi-core/feature-single/training/config-single
WORKDIR=$GITDIR
## #SBATCH --switches=1

cd $WORKDIR
source /leonardo/home/userexternal/epinning/aifs/anemoi-core/feature-single/training/venv/bin/activate

# srun anemoi-training train training=default_tendency training.run_id=393d5866510e46a8a1c74cf6012c1ab2 hardware=leonardo diagnostics.log.mlflow.run_name='16 scaling' --config-name=debug_tendency --cfg all
# srun anemoi-training train hardware=leonardo training.run_id=7e588b31f2e642fda7defe3f10c373af --config-name=debug_no-tendency
srun anemoi-training train --config-name=single23 training.run_id=0227dccda65e4caeb2cb54d256513766