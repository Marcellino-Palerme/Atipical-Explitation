#!/bin/bash
#SBATCH --job-name=multi_view
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=50G

. /softs/local/env/envconda.sh
conda activate /home/genouest/inra_umr1349/mpalerme/conda

python multi_view.py
