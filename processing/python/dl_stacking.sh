#!/bin/bash
#SBATCH --job-name=dl_stacking
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=50G

. /softs/local/env/envconda.sh
conda activate /home/genouest/inra_umr1349/mpalerme/conda

python dl_stacking.py -r ~/data_dls_r -v ~/data_dls_v
