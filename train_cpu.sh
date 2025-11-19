#!/bin/bash -l
#SBATCH --job-name=rl_train_asm
#SBATCH --output=./hpc_logs/%x.%j.out
#SBATCH --error=./hpc_logs/%x.%j.err
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=kj373
#SBATCH --mail-user=asm277@njit.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=16000M

module load bright python39
source /mmfs1/scratch/kj373/asm277/HIL-RL/python39/bin/activate
cd /mmfs1/scratch/kj373/asm277/HIL-RL/HIL-RL-Nav-Agent/NN
python example_dqn.py
