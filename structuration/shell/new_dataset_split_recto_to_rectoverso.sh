#!/bin/bash

# This script will create new dataset from a dataset only recto (symbolic link)
# to dataset rectoverso for ML (cross validation)
# Accepts four parameters: input directory, output directory
# Usage: ./new_dataset_split_recto_to_rectoverso.sh input output

# Define name of directory
dir_test="test"
dir_train="train"
dir_recto="recto"
dir_verso="verso"
lt_part=($dir_test $dir_train $dir_val)

# All split 
lt_split=$(ls -1 $1)

for split in $lt_split
do
    # Get name of directory (one by class) in input directory
    lt_classes=$(ls -1 $1/$split/$dir_test)

    for class in $lt_classes
    do
       for part in ${lt_part[@]}
       do
          # Create output directories
          mkdir -p "$2/$split/$part/$class/$dir_verso"
          mkdir -p "$2/$split/$part/$class/$dir_recto"

          # Get all file of part
          lt_images=$( ls -1 "$1/$split/$part/$class/$dir_recto" )
          for image_name in $lt_images
          do
             # Copy recto images
             cp -a "$1/$split/$part/$class/$dir_recto/$image_name" "$2/$split/$part/$class/$dir_recto"
             # Take absolu path of link of image
             image_path=$(readlink -f "$1/$split/$part/$class/$dir_recto/$image_name")
             # Determinate name of verso image
             im_verso_name=$(echo "$image_name" | sed -e "s/Recto/Verso/g")
             # Take path of image
             path=$(dirname "$image_path")
             path=$(echo "$path" | sed -e "s/recto/verso/g")
             # Create symbolic link for verso image
             ln -s "$path/$im_verso_name" "$2/$split/$part/$class/$dir_verso"
          done
       done
    done
done
