#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:25:44 2020

@author: port-mpalerme
"""

from tkinter.filedialog import askdirectory
from tools_file import file_list
from os.path import join
import numpy as np
import cv2
from PIL import Image

def max_RGB (in_rgb_image):
    a_rgb_image = in_rgb_image.copy()
    result = np.zeros(a_rgb_image.shape)
    for i in range(a_rgb_image.shape[0]):
        for j in range(a_rgb_image.shape[1]):
            my_max = np.amax(a_rgb_image[i, j])
            max_index = np.where(a_rgb_image[i, j] == my_max)
            result[i, j, max_index] = 255
    return result


def max_rgb_filter(image):
	# split the image into its BGR components
	(B, G, R) = cv2.split(image)
	# find the maximum pixel intensity values for each
	# (x, y)-coordinate,, then set all pixel values less
	# than M to zero
	M = np.maximum(np.maximum(R, G), B)
	Rb = R.copy()
	Rb[Rb < M] = 0
	Rb[Rb >= M] = 1
	G = G * Rb
	R = R * Rb
	B = B * Rb

	# merge the channels back together and return the image
	return cv2.merge([B, G, R])


def change_color(in_path_im, out_path_im):
    

    im = Image.open(in_path_im)
    data = np.array(im)
    
    r1, g1, b1 = 0, 0, 0 # Original value
    r2, g2, b2 = 0, 0, 255 # Value that we want to replace it with
    
    red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:,:,:3][mask] = [r2, g2, b2]
    
    im = Image.fromarray(data)
    im.save(out_path_im)

# take directory where are images
dir_in = askdirectory(title="in")

# take directory where save images
dir_out = askdirectory(title="out")

# take all images in directory
my_files = file_list(dir_in)

# delete background of each images
for my_file in my_files:
    try:
        # read image
        #my_image = io.imread(join(dir_in, my_file))
        change_color(join(dir_in, my_file), join(dir_out, my_file))
        image = cv2.imread(join(dir_out, my_file))
    except IOError:
        print("plouf")
        continue
    
    image = max_rgb_filter(image)
    cv2.imwrite(join(dir_out, my_file), image)