#!/bin/bash
#SBATCH --job-name=drone_rl
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# If you get access to GPU nodes later, you can switch to:
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1

echo "Job started on $(hostname)"
date

# Load conda / Anaconda environment (adjust to CSUSB's setup)
# Many clusters use: module load anaconda
# Uncomment and adapt if needed:
# module load anaconda

# Make sure conda is available in the shell
source ~/.bashrc

echo "Activating conda environment 'rl-drone'..."
conda activate rl-drone

# Go to repo root (Slurm runs from hpc/ by default if you submit there)
cd "${SLURM_SUBMIT_DIR}/.."

echo "Running HPC PPO training..."
python -m rl.train.train_ppo_hpc

echo "Job finished."
date
