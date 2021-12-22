#!/bin/bash

# This script will create new dataset from a dataset only recto (symbolic link)
# to dataset verso
# Accepts four parameters: input directory, output directory
# Usage: ./new_dataset_from_recto_to_verso.sh input output

# Define name of directory
dir_test="test"
dir_train="train"
dir_val="validation"
dir_recto="recto"
dir_verso="verso"
lt_part=($dir_test $dir_train $dir_val)

# Get name of directory (one by class) in input directory
lt_classes=$(ls -1 $1/$dir_test)

for class in $lt_classes
do
   for part in ${lt_part[@]}
   do
      # Create output directories
      mkdir -p "$2/$part/$class"

      # Get all file of part
      lt_images=$( ls -1 "$1/$part/$class" )
      for image_name in $lt_images
      do
         # Take absolu path of link of image
         image_path=$(readlink -f "$1/$part/$class/$image_name")
         # Determinate name of verso image
         im_verso_name=$(echo "$image_name" | sed -e "s/Recto/Verso/g")
         # Take path of image
         path=$(dirname "$image_path")
         # Create symbolic link for verso image
         ln -s "$path/$im_verso_name" "$2/$part/$class"
      done
   done
done