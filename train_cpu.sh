#!/bin/bash -l
#SBATCH --job-name=rl_train_asm
#SBATCH --output=rl_train.%j.out        # stdout goes here
#SBATCH --error=rl_train.%j.err         # stderr goes here
#SBATCH --partition=general             # CPU partition
#SBATCH --qos=low                       # lowest priority
#SBATCH --account=YOUR_PI_UCID          # replace with your PI's UCID
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # adjust if you need fewer/more cores
#SBATCH --time=72:00:00                 # max for standard/low QoS is 72h
#SBATCH --mem=16G                       # adjust to your needs

# ---- Environment setup ----
# Example: if you use modules + conda/venv, adapt to your setup

module load bright python39  # or whichever python module HPC recommends

# Activate venv:
source python39/bin/activate

# ---- Go to your project dir ----
cd /NN

# ---- Run your RL training ----
python example_dqn.py
