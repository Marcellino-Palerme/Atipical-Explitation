#!/bin/bash

# This script will split directory in Train, validation and test dataset
# Accepts four parameters: input directory, output directory, train size and 
# validation size
# Usage: ./Split_Train_Valid_Test.sh input output train validation

# Verify train size between 0 and 1
if (( $(echo "$3 >= 1" | bc -l) ))
then
   echo "Train_size too great (>=1)"
   exit 1
fi

if (( $(echo "$3 <= 0" | bc -l) ))
then
   echo "Train_size too lower (<=0)"
   exit 2
fi

# Verify train size between 0 and 1
if (( $(echo "$4 >= 1" | bc -l) ))
then
   echo "Validation_size too great (>=1)"
   exit 3
fi

if (( $(echo "$4 <= 0" | bc -l) ))
then
   echo "Validation_size too lower (<=0)"
   exit 4
fi

# Verify if it stay images for the test
if (( $(echo "$4 + $3 >= 1" | bc -l) ))
then
   echo "No image for test"
   exit 5
fi

answer=0

function round()
{
   # Take decimal part
   part_dec=$(echo "$1 % 1" | bc )
   # Take inter part
   part_int=$(bc <<<$1-$part_dec)
   # Verify if decimal part greater or equal 0.5
   if (( $(echo "$part_dec >= 0.5" | bc ) ))
   then 
      answer=$( echo "$part_int + 1"| bc )
   else
      answer=$( echo "$part_int"| bc )
   fi
   echo $answer
}

# Get name of directory (one by class) in input directory
lt_classes=$(ls -1 $1) 

for class in $lt_classes
do
   # Create output directories
   mkdir -p "$2/train/$class/recto"
   mkdir -p "$2/validation/$class/recto"
   mkdir -p "$2/test/$class/recto"
   mkdir -p "$2/train/$class/verso"
   mkdir -p "$2/validation/$class/verso"
   mkdir -p "$2/test/$class/verso"
   # get a shuffle list of image of class
   lt_images=($(ls -1 "$1/$class/recto" | shuf))
   # get number file in directory
   tot_size=${#lt_images[*]}
   
   # Calculate number of elements for train
   train_size=$( echo "$3 * $tot_size"| bc )
   round $train_size
   train_size=$answer
   
   # Calculate number of elements for valid
   valid_size=$( echo "$4 * $tot_size"| bc )
   round $valid_size
   valid_size=$answer
   
   # Calculate last index for validation
   valid_index=$( echo "$train_size+$valid_size"| bc )
   for index in ${!lt_images[*]}
   do
      str=${lt_images[$index]}
      img_verso=$(echo "$str" | sed -e "s/Recto/Verso/g") 
      if (( $(echo "$index < $train_size" | bc -l) )) 
      then
         # add symbolic link of image in train
         ln -s "$1/$class/${lt_images[$index]}" "$2/train/$class/recto"
         echo "1"
         ln -s "$1/$class/$img_verso" "$2/train/$class/verso"
      elif (( $(echo "$index < $valid_index" | bc -l) )) 
      then
         # add symbolic link of image in validation
         ln -s "$1/$class/${lt_images[$index]}" "$2/validation/$class/recto"
         ln -s "$1/$class/$img_verso" "$2/validation/$class/verso"
      elif (( $(echo "$index >= $valid_index" | bc -l) ))
      then
         # add symbolic link of image in test
         ln -s "$1/$class/${lt_images[$index]}" "$2/test/$class/recto"
         ln -s "$1/$class/$img_verso" "$2/test/$class/verso"
      fi
   done
     
done
 

