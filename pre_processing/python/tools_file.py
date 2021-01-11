#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provide several function to manipulate files and directories
"""

# os related (directory and file stuff)

from os import listdir, makedirs, rmdir, remove
from os.path import isfile, join, exists, splitext
from skimage import io
import skimage.transform as tr

__all__ = ["file_list", "create_directory", "file_list_ext", "rm_directory"]


def file_list(directory):
    """!@brief
        give list of files in a directory

        @param directory (str)
            directory where search files

        @return (list)
            list of files
    """
    onlyfiles = [f for f in listdir(directory)
                 if isfile(join(directory, f))]
    onlyfiles.sort()

    return onlyfiles


def file_list_ext(directory, ext):
    """!@brief
        give list of files in a directory of one type

        @param directory: str
            directory where are the files
        @param ext: str
            extension of kept files (with .)

        @return (list of str)
            list of name of file
    """
    lt_file = file_list(directory)
    return [filename for filename in lt_file
            if splitext(filename)[1] == '.' + ext]


def create_directory(path):
    """!@brief
        create a_directory

        @param path (str)
            name of directory or complete path
    """
    if not exists(path):
        try:
            makedirs(path)
        except IOError:
            pass


def rm_directory(path):
    """!@brief
        delete a directory

        @param path (str)
            name of directory or complete path
    """
    if exists(path):
        try:
            for rm_file in file_list(path):
                remove(path + '/' + rm_file)
            rmdir(path)
        except IOError:
            pass


def rescale_im(dir_in, ratio, dir_out=None):
    """!@brief
        change size of image

        @param dir_in (str)
            directory of image
        @param ratio (float)
           ratio to change image (ex: 2 => *2; 0.5 => /2)
        @param dir_out (str)
           directory where save new image if None then resized image replace
           origin image
    """
    # take all files
    lt_file = file_list(dir_in)

    # Verify if output directory define
    if dir_out is None:
        dir_out = dir_in
    else:
        # Create output directory if it is necessary
        create_directory(dir_out)

    for my_file in lt_file:
        # Verify if the file is a image
        try:
            my_image = io.imread(dir_in + "/" + my_file)
        except IOError:
            pass
        else:
            # Resize image
            my_image = tr.rescale(my_image, ratio)
            # Save resized image
            io.imsave(dir_out + "/" + my_file,
                      my_image,
                      plugin='pil')

if __name__ == "__main__":
    rescale_im("Corpus_search", 0.5, "corpus_search_resize")
