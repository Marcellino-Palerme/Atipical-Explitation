#!/bin/bash
#SBATCH --job-name=tl_atipical
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=50G
#SBATCH --output=.report/$(date "+%Y%m%d_%H%M%S")results_tl_atipical_B4.txt

. /softs/local/env/envconda.sh
conda activate /home/genouest/inra_umr1349/mpalerme/conda
python tl_atipical.py  
