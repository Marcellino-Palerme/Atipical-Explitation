#!/bin/bash
#SBATCH --job-name=validation
#SBATCH --gres=gpu:1
#SBATCH -p gpu

.  /softs/local/env/envpython-3.8.5.sh
virtualenv ~/my_deep_validation
. ~/my_deep_validation/bin/activate
pip install --upgrade pip
pip install tensorflow
pip install keras
pip install numpy
pip install scikit-learn
python confusion_matrix_deep.py
