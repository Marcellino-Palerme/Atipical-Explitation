#!/bin/bash

# This script will create new dataset from a dataset only recto (symbolic link)
# to dataset recto/verso
# Accepts four parameters: input directory, output directory
# Usage: ./new_dataset_from_recto_to_recto_verso.sh input output

# Define name of directory
dir_test="test"
dir_train="train"
dir_val="validation"
dir_recto="recto"
dir_verso="verso"

# Get name of directory (one by class) in input directory
lt_classes=$(ls -1 $1/$dir_test) 

for class in $lt_classes
do
   # Create output directories
   mkdir -p "$2/$dir_test/$class/$dir_recto"
   mkdir -p "$2/$dir_test/$class/$dir_verso"
   mkdir -p "$2/$dir_train/$class/$dir_recto"
   mkdir -p "$2/$dir_train/$class/$dir_verso"
   mkdir -p "$2/$dir_val/$class/$dir_recto"
   mkdir -p "$2/$dir_val/$class/$dir_verso"
   
   # Get all file of test part
   lt_images=$( ls -1 "$1/$dir_test/$class" )

   for image_name in $lt_images
   do
      # Take absolu path of link of image
      image_path=$(readlink -f "$1/$dir_test/$class/$image_name")
      # Create symbolic link in new directory
      ln -s $image_path "$2/$dir_test/$class/$dir_recto"
      # Determinate name of verso image
      im_verso_name=$(echo "$image_name" | sed -e "s/Recto/Verso/g")
      # Take path of image
      path=$(dirname "$image_path")
      # Create symbolic link for verso image
      ln -s "$path/$im_verso_name" "$2/$dir_test/$class/$dir_verso"
   done

   # Get all file of train part
   lt_images=$( ls -1 "$1/$dir_train/$class" )

   for image_name in $lt_images
   do
      # Take absolu path of link of image
      image_path=$(readlink -f "$1/$dir_train/$class/$image_name")
      # Create symbolic link in new directory
      ln -s $image_path "$2/$dir_train/$class/$dir_recto"
      # Determinate name of verso image
      im_verso_name=$(echo "$image_name" | sed -e "s/Recto/Verso/g")
      # Take path of image
      path=$(dirname "$image_path")
      # Create symbolic link for verso image
      ln -s "$path/$im_verso_name" "$2/$dir_train/$class/$dir_verso"
   done
   
   # Get all file of validation part
   lt_images=$( ls -1 "$1/$dir_val/$class" )

   for image_name in $lt_images
   do
      # Take absolu path of link of image
      image_path=$(readlink -f "$1/$dir_val/$class/$image_name")
      # Create symbolic link in new directory
      ln -s $image_path "$2/$dir_val/$class/$dir_recto"
      # Determinate name of verso image
      im_verso_name=$(echo "$image_name" | sed -e "s/Recto/Verso/g")
      # Take path of image
      path=$(dirname "$image_path")
      # Create symbolic link for verso image
      ln -s "$path/$im_verso_name" "$2/$dir_val/$class/$dir_verso"
   done
done