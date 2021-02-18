#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:58:47 2021

@author: mpalerme
"""
from tkinter.filedialog import askdirectory
from tools_file import file_list, create_directory
import cv2
from maxRGB import change_color, max_rgb_filter
from os.path import join, splitext, basename, exists
from os import remove
from PIL import Image
import logging
from skimage import morphology, io
from skimage import measure, color
from skimage.morphology import square, ball
import numpy as np
from del_bg import delete_bg
from multiprocessing import Pool
import uuid

col = 1120
row = 560

def func_info_residu(img_res_wout_bg):
    """!@brief
        return information on Rapeseed residu

        @param img_res_wout_bg : array 2d
            Image of one Rapeseed residu without background

        @return (list)
            list of region and their information
    """
    logging.info("IN")
    logging.debug("%s", locals())

    # obtain a black and white image
    temp_img = img_res_wout_bg[:, :, 0].copy()
    a_no_zero = img_res_wout_bg[:, :, 0].nonzero()
    a_no_zero = a_no_zero + img_res_wout_bg[:, :, 1].nonzero()
    a_no_zero = a_no_zero + img_res_wout_bg[:, :, 2].nonzero()

    # create black and white image
    for my_index in range(len(a_no_zero[0])):
        absc = a_no_zero[0][my_index]
        ordo = a_no_zero[1][my_index]
        temp_img[absc][ordo] = 255

    # delete black points in residu
    temp_img = morphology.closing(temp_img, square(5))

    # delete residu dust near residu
    temp_img = morphology.erosion(temp_img, square(20))

    # Find regions and give informations about them
    temp_img = morphology.label(temp_img)
    list_rp_residu = measure.regionprops(temp_img)

    logging.info("OUT")
    # Keep only 5 information about each region
    return map(func_map_regionlist, list_rp_residu)


def func_map_regionlist(rp_region):
    """!@brief
        Keep only 2 information about a region
                 - area
                 - centroid


        @param rp_region : skimage.measure.regionprops
            RegionProperties (from skimage.measure.regionprops)

        @return (dict)
            a dictionary with 2 information
    """
    return {"area": rp_region.area, "centroid": rp_region.centroid}

def my_closing(im_one_channel):
    return morphology.closing(im_one_channel, square(100))


def center(my_file):
    # read image
    try:
        or_im = Image.open(join(dir_in, my_file))
    except IOError:
        return 0 

    # keep name file without extension
    name = splitext(basename(my_file))[0]
    name_temp = str(uuid.uuid4()) + ".bmp"
    name_close = str(uuid.uuid4()) + ".bmp"
    name_max = str(uuid.uuid4()) + ".tiff"
    
    # Image smaller bounding-box case
    if or_im.size[0] < col and or_im.size[1] < row:
        # Create à black image
        temp = Image.new("RGB", (col, row))
        # Paste image
        temp.paste(or_im)
        # Save image
        temp.save(join(dir_out, name + ".bmp"))
        # pass next image
        return 0
    # Column is too small
    elif or_im.size[0] < col:
         # Create à black image
        temp = Image.new("RGB", (col + 10, or_im.size[1]))
        # Paste image
        temp.paste(or_im)
        # Save image
        temp.save(join(dir_in, name_temp))
        # Continue work with nex image
        my_file = name_temp
        or_im = Image.open(join(dir_in, my_file))
    # Row is too small
    elif or_im.size[1] < row:
         # Create à black image
        temp = Image.new("RGB", (or_im.size[0], row + 10))
        # Paste image
        temp.paste(or_im)
        # Save image
        temp.save(join(dir_in, name_temp))
        # Continue work with nex image
        my_file = name_temp
        or_im = Image.open(join(dir_in, my_file))

    # Closing 
    image = io.imread(join(dir_in, my_file), plugin='matplotlib')
    image_c = image.copy()
    image_c[:, :, 0] = morphology.closing(image[:, :, 0], square(100))
    image_c[:, :, 1] = morphology.closing(image[:, :, 1], square(100))
    image_c[:, :, 2] = morphology.closing(image[:, :, 2], square(100))

    io.imsave(join(dir_in, name_close), image_c, plugin="pil")
    
    # max RGB
    #change_color(join(dir_in, my_file), join(dir_out, "tmp.tiff"))
    image = cv2.imread(join(dir_in, name_close))
    image = max_rgb_filter(image)
    cv2.imwrite(join(dir_out, name_max), image)
    im_maxRGB = io.imread(join(dir_out, name_max), plugin='matplotlib')
    # Find each residu in image
    residu_maxRGB = func_info_residu(im_maxRGB)

    # Double otsu
    im_otsu = io.imread(join(dir_in, name_close), plugin='matplotlib')
    im_otsu = delete_bg(im_otsu, 3)
    im_otsu = delete_bg(im_otsu, 1)

    # io.imshow(im_otsu)
    
    # Find each residu in image
    residu_otsu = func_info_residu(im_otsu)

    # verify if find residu
    residu = list(residu_otsu) + list(residu_maxRGB)
    if len(residu) == 0 :
        # Take image's center
        center = [or_im.size[1]//2, or_im.size[0]//2]
    else:
        # Sort residus by size
        residu = sorted(residu, key=lambda k: k["area"])
        # Keep the bigger residu
        residu = residu[-1]
        # Take  residu's center
        center = [int(np.around(x)) for x in residu["centroid"]]

    # center too close to border
    if center[0] - (row//2) < 0:
        center[0] -= center[0] - (row//2)
    if center[1] - (col//2) < 0:
        center[1] -= center[1] - (col//2)
    if center[0] + (row//2) - or_im.size[1] > 0:
        center[0] = center[0] - (center[0] + (row//2) - or_im.size[1])
    if center[1] + (col//2) - or_im.size[0] > 0:
        center[1] -= center[1] + (col//2) - or_im.size[0]
    
    or_im = io.imread(join(dir_in, my_file), plugin='matplotlib')
    io.imsave(join(dir_out, name + ".bmp"), 
              or_im[center[0]- (row//2):center[0] + (row//2),
                    center[1] - (col//2):center[1] + (col//2), :],
              plugin="pil")
    
    # Delete temporary files
    if exists(join(dir_in, name_temp)):
        remove(join(dir_in, name_temp))
    if exists(join(dir_in, name_close)):
        remove(join(dir_in, name_close))
    if exists(join(dir_out, name_max)):
        remove(join(dir_out, name_max))
    return 0

if __name__ == "__main__":
    # take directory where are images
    dir_in = askdirectory(title="in")
    
    # take directory where save images
    dir_out = askdirectory(title="out")
    
    # take all images in directory
    my_files = file_list(dir_in)

    # Create out directory
    create_directory(dir_out)

    pool = Pool(processes=18)
    pool.map(center, my_files)       
