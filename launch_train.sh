#!/bin/bash
#SBATCH --job-name=slml_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpushort
#SBATCH --time=1:0:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$HOME/SLML"
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Your venv seems to be nested here:
source "$HOME/SLML/SLML/bin/activate"

which python
python --version

python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY

python train.py