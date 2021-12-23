#!/bin/bash

# This script will modify name and symlink
# from *balanced_0.bmp to *balanced.bmp
# Accepts one parameter: input directory
# Usage: ./modify_name_n_symlink.sh input

# Get all file
lt_files=$(find -L $1 -type f,l)

for file in $lt_files
do
   # Modify file name
   new_name=$(echo "$file" | sed -e 's/balanced_[0-99]/balanced/g')
   if [ $new_name != $file ]
   then
      mv $file $new_name
   fi
   # Take absolu path of link
   symlink=$(readlink $new_name)
   # Verify if there is a symbolic link
   if [ ! -z "$symlink" ]
   then
      # modify file name of symlink
      symlink=$(echo "$symlink" | sed -e 's/balanced_[0-99]/balanced/g')
      symlink=$(echo "$symlink" | sed -e 's/\/\//\//g')
      # modify symbolic link
      ln -s -f $symlink $new_name
   fi
done