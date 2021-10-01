#!/bin/bash
#SBATCH --job-name=B5_atipical
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=50G

. /softs/local/env/envconda.sh
conda activate /home/genouest/inra_umr1349/mpalerme/conda
python tl_plant_leave.py -i /home/genouest/inra_umr1349/mpalerme/result/atipical -o /home/genouest/inra_umr1349/mpalerme/result/atipical/B5 -5
