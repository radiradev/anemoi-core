#!/bin/bash

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=450GB
#SBATCH --time=48:00:00
#SBATCH --account=ecaifs
#SBATCH --output=autoencoder-huber-ml137-o32-512.%j.out
#SBATCH --job-name=autoencoder-huber-ml137-o32-512


# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
# export NCCL_DEBUG=TRACE
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_LAUNCH_BLOCKING=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export NCCL_IB_DISABLE=0 # Enable InfiniBand if available
# export TORCH_NCCL_BLOCKING_WAIT=0
# export NCCL_BUFFSIZE=8388608
# export NCCL_NSOCKS_PERTHREAD=4
# export NCCL_SOCKET_NTHREADS=2

# generic settings
VENV=anemoi-autoencoder
GITDIR=/home/syma/ANEMOI-CORE/autoencoder-dev
WORKDIR=$GITDIR

cd $WORKDIR
module load python3/new
module load nvidia/24.11
source /home/$USER/venvs/$VENV/bin/activate

srun anemoi-training train --config-name=autoencoder_huber
