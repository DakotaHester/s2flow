#!/bin/bash
#SBATCH --job-name=s2flow_reg_eval
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/dhester/s2flow/logs/s2flow_sr_train_regression/slurm_eval_out.log
#SBATCH --error=/home/dhester/s2flow/logs/s2flow_sr_train_regression/slurm_eval_out.log

cd /home/dhester/s2flow

./.venv/bin/python -m s2flow.cli --config configs/s2flow-sreval_regression.yaml
