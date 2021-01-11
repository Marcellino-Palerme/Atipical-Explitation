#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:43:25 2020

@author: mpalerme

Concatenate recto and verso in same image
"""
from tkinter.filedialog import askdirectory
from tools_file import create_directory
import glob
import cv2
from os.path import join, basename, splitext

# take directory where are images
dir_in = askdirectory(title="in")

# take directory where save images
dir_out = askdirectory(title="out")

# Create directory to save images
create_directory(dir_out)

# take all images in directory
recto = sorted(glob.glob(dir_in + "/*ecto*.tiff"))
verso = sorted(glob.glob(dir_in + "/*erso*.tiff"))

for index in range(len(recto)):
    # take recto image
    im_recto = cv2.imread(recto[index])
    # resize recto image
    im_recto = cv2.resize(im_recto, (224,112))
    # take verso image
    im_verso = cv2.imread(verso[index])
    # Resize verso image
    im_verso = cv2.resize(im_verso, (224,112))
    # Create one image to concatenate recto and verso images
    im_v = cv2.vconcat([im_recto, im_verso])

    # Save concatenate image in bmp format
    filename = splitext(basename(recto[index]))[0] + ".bmp"
    cv2.imwrite(join(dir_out, filename), im_v)
