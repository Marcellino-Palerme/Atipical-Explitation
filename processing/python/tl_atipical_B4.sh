#!/bin/bash

#Create temporaly file
temp_file=$(date "+%Y%m%d_%H%M%S")_temp.txt

#Add all line to run deep learning
echo '#!/bin/bash' >> $temp_file
echo \#SBATCH --job-name=tl_atipical_B4_$(date "+%Y%m%d_%H%M%S") >> $temp_file
echo '#SBATCH --gres=gpu:1' >> $temp_file
echo '#SBATCH -p gpu' >> $temp_file
echo '#SBATCH --mem=50G' >> $temp_file
echo \#SBATCH --output=./report/$(date "+%Y%m%d_%H%M%S")_results_tl_atipical_B4.txt >> $temp_file

echo . /softs/local/env/envconda.sh >> $temp_file
echo conda activate /home/genouest/inra_umr1349/mpalerme/conda >> $temp_file
echo python tl_atipical_B4.py  >> $temp_file

#Run temporaly file
sbatch $temp_file
