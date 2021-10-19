#!/bin/bash

my_date=$(date "+%Y%m%d_%H%M%S")

# Define structure used
struc="VGG16"

#Create temporaly file
temp_file="${my_date}_temp.txt"

#Add all line to run deep learning
echo '#!/bin/bash' >> $temp_file
echo "#SBATCH --job-name=tl_atipical_${struc}_${my_date}" >> $temp_file
echo '#SBATCH --gres=gpu:1' >> $temp_file
echo '#SBATCH -p gpu' >> $temp_file
echo '#SBATCH --mem=50G' >> $temp_file
echo "#SBATCH --output=./report/${my_date}_tl_atipical_${struc}/${my_date}_results_tl_atipical_${struc}.txt" >> $temp_file

echo . /softs/local/env/envconda.sh >> $temp_file
echo conda activate /home/genouest/inra_umr1349/mpalerme/conda >> $temp_file
echo "python tl_atipical_XX.py ${my_date} ${struc}" >> $temp_file

#Create directory of report
mkdir -p "./report/${my_date}_tl_atipical_${struc}"

#Run temporaly file
sbatch $temp_file
