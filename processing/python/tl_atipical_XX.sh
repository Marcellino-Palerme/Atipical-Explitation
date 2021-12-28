#!/bin/bash

#SBATCH --job-name=tl_atipical_XX
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=50G

my_date=$(date "+%Y%m%d_%H%M%S")

#Create directory of report
mkdir -p "./report/${my_date}_tl_atipical_XX"

. /softs/local/env/envconda.sh
conda activate /home/genouest/inra_umr1349/mpalerme/conda
python tl_atipical_XX.py ${my_date} B3
python tl_atipical_XX.py ${my_date} B4
python tl_atipical_XX.py ${my_date} B5
python tl_atipical_XX.py ${my_date} B6
python tl_atipical_XX.py ${my_date} VGG16

