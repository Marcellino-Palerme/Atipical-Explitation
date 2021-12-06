#!/bin/bash
#SBATCH --job-name=augmented_images
#SBATCH --mem=50G

. /softs/local/env/envconda.sh
conda activate /home/genouest/inra_umr1349/mpalerme/conda
python augmented_images.py -i /home/genouest/inra_umr1349/mpalerme/reduc_atipical --rv
