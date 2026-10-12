#!/bin/bash
#SBATCH --job-name=s2flow_reg_train
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --output=/home/dhester/s2flow/logs/s2flow_sr_train_regression/slurm_out.log
#SBATCH --error=/home/dhester/s2flow/logs/s2flow_sr_train_regression/slurm_out.log

cd /home/dhester/s2flow

./.venv/bin/python -m s2flow.cli --config configs/s2flow-srtrain_regression.yaml