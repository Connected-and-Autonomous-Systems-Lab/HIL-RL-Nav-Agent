#!/bin/bash -l
#SBATCH --job-name=rl_train_asm
#SBATCH --output=rl_train_asm.%j.out        # stdout goes here
#SBATCH --error=rl_train.%j.err         # stderr goes here
#SBATCH --partition=general             # CPU partition
#SBATCH --qos=low                       # lowest priority
#SBATCH --account=kj373                 #  PI's UCID
#SBATCH --mail-user asm277@njit.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00                 # max for standard/low QoS is 72h
#SBATCH --mem-per-cpu=8000M                       

# ---- Environment setup ----
# Example: if you use modules + conda/venv, adapt to your setup

module load bright python39  # or whichever python module HPC recommends

# Activate venv:
source ../python39/bin/activate

# ---- Go to your project dir ----
cd /NN

# ---- Run your RL training ----
python example_dqn.py
