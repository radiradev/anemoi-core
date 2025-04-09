#!/bin/bash
#SBATCH --qos=ng
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --account=ecaifs
#SBATCH --mem=240G
#SBATCH --time=47:00:00
#SBATCH --output=slurm/output-%j.out
#set -x

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

GITDIR=/home/daep/PERM/projects/anemoi-core/feature-snow/training/config-gab
WORKDIR=$GITDIR
cd $WORKDIR

export CUDA_LAUNCH_BLOCKING=1

# generic settings
#VENV=/home/daep/PERM/projects/anemoi-core/feature-scaler-split/training/venv
VENV=/home/daep/PERM/projects/anemoi-core/feature-snow/training/venv

module load python3
source ${VENV}/bin/activate
# export PATH="$HOME/.local/bin:$PATH"

#srun anemoi-training train --config-name=debug_o96-std-tendency hardware=slurm
srun anemoi-training train --config-name=training_6h_transformer_mse_diagsm
