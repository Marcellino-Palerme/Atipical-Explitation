#!/bin/bash

# This script will modify symlink
# from *balanced_0.bmp to *balanced.bmp
# Accepts one parameter: input directory
# Usage: ./modify_symlink.sh input

# Get all file
lt_files=$(find -L $1 -type f)

for file in $lt_files
do
   # Take absolu path of link
   symlink=$(readlink -f $file)
   # modify file name of symlink
   symlink=$(echo "$symlink" | sed -e 's/balanced_[0-99]/balanced/g')
   symlink=$(echo "$symlink" | sed -e 's/\/\//\//g')
   # modify symbolic link
   ln -s -f $symlink $file
done