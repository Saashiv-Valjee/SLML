#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=asl_tcn
#SBATCH --output=logs/asl_tcn_%j.out
#SBATCH --error=logs/asl_tcn_%j.err

mkdir -p logs

module load cuda

source ~/SLML/SLML/bin/activate

echo "Host:"
hostname

echo "Python:"
which python

echo "nvidia-smi:"
nvidia-smi

echo "TensorFlow GPUs:"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

python train.py