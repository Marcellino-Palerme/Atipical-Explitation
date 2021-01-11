#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    
"""


from tkinter.filedialog import askdirectory
from tools_file import file_list, create_directory
from os.path import join
from shutil import move
from PIL import Image
from PIL.ExifTags import TAGS


# take directory where are images
dir_in = askdirectory(title="in")

# take directory where save images with different orientation
dir_out = askdirectory(title="out")

# Create directory to save images
create_directory(dir_out)

# take all images in directory
my_files = file_list(dir_in)

# Extract all images with a different orientation
for my_file in my_files:
    try:
        # read image
        img = Image.open(join(dir_in, my_file))
    except IOError:
        continue
    # build reverse dicts
    _TAGS_r = dict(((v, k) for k, v in TAGS.items()))
    if _TAGS_r is None:
       continue
    exifd = img._getexif()  # as dict
    if exifd is None:
       continue
    if(exifd[_TAGS_r["Orientation"]] == 1):
       move(join(dir_in, my_file), join(dir_out, my_file))
    
    	
