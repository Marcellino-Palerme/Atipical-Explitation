#!/bin/bash
#SBATCH --job-name=photo
#SBATCH --cpus-per-task=30
#SBATCH --mem=50G

.  /softs/local/env/envpython-3.8.5.sh
virtualenv ~/my_photo_env
. ~/my_photo_env/bin/activate
pip install --upgrade pip
pip install numpy
pip install -r /home/genouest/inra_umr1349/mpalerme/atipical-exploi/requirements_py_3_9.txt 
python center.py
rm -rf ~/my_photo_env
