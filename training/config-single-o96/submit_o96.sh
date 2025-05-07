#!/bin/bash
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --account=ecdestin  # ecrdasca  # ecaifs
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/single01-%j.out
#set -x

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export CUDA_LAUNCH_BLOCKING=1
# export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=TRACE
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# generic settings

GITDIR=/perm/daep/projects/anemoi-core/feature-single01/training/config-single
WORKDIR=$GITDIR
cd $WORKDIR

export CUDA_LAUNCH_BLOCKING=1

# generic settings
VENV=/home/daep/PERM/projects/anemoi-core/feature-single01/training/venv

module load python3
source ${VENV}/bin/activate

srun anemoi-training train --config-name=single
#srun anemoi-training train --config-name=single_snow
