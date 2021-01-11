#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provide several function to manipulate files and directories
"""

# os related (directory and file stuff)

from os import listdir, makedirs, rmdir, remove
from os.path import isfile, join, exists, splitext
import shutil


__all__ = ["file_list", "create_directory", "file_list_ext", "rm_directory",
           "copy"]


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

def copy(src, dst, overwrite=False):
    """!@brief
        copy a directory or a file

        @param src: str
            source path of file or directory
        @param dst: str
            destination path of file or directory
        @param overwrite: bool
            overwrite
        @return: int
            0 -> OK
            1 -> No right to copy
            2 -> src not exist or no overwrite
    """
    if exists(src) and (not exists(dst) or overwrite):
        try:
            shutil.copy2(src, dst)
        except IOError:
            # Copy impossible no right  
            return 1
    else:
        # Copy impossible src not exist or no overwrite
        return 2
    # Copy Ok
    return 0
