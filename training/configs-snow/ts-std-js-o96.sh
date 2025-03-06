#!/bin/bash

#SBATCH -A EUHPC_E04_053
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/aifs-o96-std-tendency.out.%j

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


GITDIR=/leonardo/home/userexternal/epinning/aifs/anemoi-core/feature-snow/training/configs-snow
WORKDIR=$GITDIR

cd $WORKDIR
source /leonardo/home/userexternal/epinning/aifs/anemoi-core/feature-scaler-split/training/venv/bin/activate

#srun anemoi-training train training=default_tendency training.run_id=393d5866510e46a8a1c74cf6012c1ab2 hardware=leonardo diagnostics.log.mlflow.run_name='16 scaling' --config-name=debug_tendency --cfg all
srun anemoi-training train hardware=leonardo --config-name=debug_o96-std-tendency
