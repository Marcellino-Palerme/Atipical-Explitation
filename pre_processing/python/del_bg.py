#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Replace the background by black
'''

import gc
import logging
from skimage import io, color, filters
import numpy as np
from tkinter.filedialog import askdirectory
from tools_file import file_list
from os.path import join, splitext


def delete_bg(a_rgb_image, ref_threshold=0):
    """!@brief
        delete RGB image's background
        To delete back

        @param a_rgb_image : array 3D of shape (.., .., 3).
            Rbg image
        @param ref_threshold (int)
                0 - where is background undeterminate
                1 - less element than background
                2 - more element than background
                3 - background is pixel inferior Otsu threshold
                4 - background is pixel superior Otsu threshold

        @return (array 3D of shape (.., .., 3))
            Rgb image without background
    """
    logging.info("IN")
    logging.debug("%s", locals())

    # Copy to work on copy of array and not on array
    a_rgb_image = a_rgb_image.copy()

    # change RGB image in HSV
    hsv_image = color.rgb2hsv(a_rgb_image)

    # otsu filtering value on hue
    im_thres_otsu = filters.threshold_otsu(hsv_image[:, :, 0])

    binary_im = hsv_image[:, :, 0] <= im_thres_otsu

    nb_element = np.unique(binary_im, return_counts=True)[1][1]
    # case background is pixel superior Otsu threshold
    if ref_threshold == 4:
        binary_im = hsv_image[:, :, 0] > im_thres_otsu
    # case less element than background
    elif ref_threshold == 1:
        if nb_element > (hsv_image[:, :, 0].size - nb_element):
            binary_im = hsv_image[:, :, 0] > im_thres_otsu
    # case more element than background
    elif ref_threshold == 2:
        if nb_element <= (hsv_image[:, :, 0].size - nb_element):
            binary_im = hsv_image[:, :, 0] > im_thres_otsu

    binary_im.dtype = "uint8"

    # create new image without background
    a_rgb_image[:, :, 0] = a_rgb_image[:, :, 0] * binary_im
    a_rgb_image[:, :, 1] = a_rgb_image[:, :, 1] * binary_im
    a_rgb_image[:, :, 2] = a_rgb_image[:, :, 2] * binary_im
    del binary_im
    gc.collect()
    logging.info("OUT")
    return a_rgb_image


# take directory where are images
dir_in = askdirectory()

# take all images in directory
my_files = file_list(dir_in)

# delete background of each images
for my_file in my_files:
    try:
        # read image
        my_image = io.imread(join(dir_in, my_file))
    except IOError:
        continue
    # delete background
    # we suppose there are background is pixel inferior Otsu threshold
    my_image = delete_bg(my_image, 3)
    # keep name of file without extension
    name_wo_ext = splitext(my_file)[0]
    # Save image in tiff to background isn't spread with element
    io.imsave(join(dir_in, name_wo_ext) + ".tiff", my_image)
    
    	
