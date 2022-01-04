#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:53:01 2021

@author: mpalerme

Create two directory, recto and verso
- dl
+- train
++- Alt
++- Big
++- ...
+- validation
++- Alt
++- Big
++- ...
- stacking
+- 0
++- train
+++- Alt
+++- Big
+++- ...
++- test
++- train
+++- Alt
+++- Big
+++- ...
+- 1
...

"""

import argparse
import os.path as op
import glob
from os import symlink
import sklearn.model_selection as ms
import tools_file as tf
import numpy as np

##############################################################################
### Constants
##############################################################################
LT_CLASS = ["Alt", "Big", "Mac", "Mil", "Myc", "Pse", "Syl"]
RECTO = "recto"
VERSO = "verso"
TRAIN = "train"
TEST = "test"
CST_DL = "dl"
CST_STACK = "stacking"
VAL = "validation"
LAB = "label"


##############################################################################
### Additional function
##############################################################################
def create_struct_dir(path, nb_split):
    """
    create directory structure for spliting

    Parameters
    ----------
    path : str
        path where create struct.
    nb_split : int
        Number of split for stacking part.

    Returns
    -------
    None.

    """
    # Take absolue path
    abs_path = op.abspath(op.expanduser(path))
    dl_path = op.join(abs_path, CST_DL)
    stack_path = op.join(abs_path, CST_STACK)

    for sympt in LT_CLASS:
        # Create directory of deep learning part
        for part in [TRAIN, VAL]:
            tf.create_directory(op.join(dl_path, part, sympt))

        # Create directory of stacking part
        for split in range(nb_split):
            for part in [TRAIN, TEST]:
                tf.create_directory(op.join(stack_path, str(split), part,
                                            sympt))

def fit_directory(dir_out, face, dataset, lt_index):
    """
    Add symbolic link in directory

    Parameters
    ----------
    dir_out : str
        path of part to fill.
    face : str
        recto or verso
    dataset : dictionary of list
        path and symptom of all images.
    lt_index : list of int
        index of image keep to fill the directory.

    Returns
    -------
    None.

    """
    # Take absolue path
    abs_path = op.abspath(op.expanduser(dir_out))
    for index in lt_index:
        # Get name of symptom
        sympt = dataset[LAB][index]
        # Get path of origin image
        img_path = dataset[face][index]
        # Extract name of image
        img_name = op.basename(img_path)
        # Create symbolic link
        symlink(img_path, op.join(abs_path, sympt, img_name))


##############################################################################
### Function to verify argument input
##############################################################################
class PercentAction(argparse.Action):
    """
        manage the min and max of percentage of train
        return Decimal object
    """
    def __call__(self, parser, namespace, values, option_string=None):
        # Verify value strictly higher than 0
        if values <= 0:
            parser.error("Proportion have to be higher than 0")
        # Verify value strictly lower than 1
        if values >= 1:
            parser.error("Proportion have to be lower than 1")

        setattr(namespace, self.dest, values)

class NumberSplitAction(argparse.Action):
    """
        manage the min of number of split
    """
    def __call__(self, parser, namespace, values, option_string=None):
        # Verify if value is strictly higher than 1
        if values <= 1:
            parser.error("Number of split have to be higher than 1")

        setattr(namespace, self.dest, values)

###############################################################################
### Manage arguments input
###############################################################################
def arguments ():
    """
    manage input arguments

    Returns
    -------
    namespace

    """
    parser = argparse.ArgumentParser()

    # Add argument for source directory
    parser.add_argument('-i', '--dir_in', type=str, help="source directory",
                        required=True, dest='dir_in')

    # Add argument for output directory
    parser.add_argument('-r', '--dir_rec', type=str,
                        help="output recto directory",
                        required=True, dest='dir_rec')

    # Add argument for output directory
    parser.add_argument('-v', '--dir_ver', type=str,
                        help="output verso directory",
                        required=True, dest='dir_ver')

    # Add argument for proportion of image for training deep learning
    parser.add_argument('-t', '--train',
                        help="proportion of image for training deep learning",
                        default=0.6, dest='train', action=PercentAction)

    # Add argument for proportion of image for validation deep learning
    parser.add_argument('-c', '--validation',
                        help="proportion of image for validation deep learning",
                        default=0.1, dest='val', action=PercentAction)

    # Add argument for number split for k-fold for cross-validation of stacking
    parser.add_argument('-s', '--split',
                        help="number split for cross-validation of stacking",
                        default=5, dest='split', action=NumberSplitAction,
                        type=int)

    # Take all arguments
    return parser.parse_args()

##############################################################################
### Main function
##############################################################################
def run():
    """
    just run script

    Returns
    -------
    None.

    """
    args = arguments()

    # Verify stay proportion for stacking
    if (args.train + args.val) >= 1 :
        raise Exception("Proportion's sum of train and validation is higher than 1")

    # Calculate proportion for stacking
    pp_stack = 1-args.train-args.val

    # Get absolu path of input directory
    abs_input = op.abspath(op.expanduser(args.dir_in))

    # Create list of all images
    all_img = {RECTO:[], VERSO:[], LAB:[]}
    for sympt in LT_CLASS:
        # Get recto image
        lt_recto = sorted(glob.glob(op.join(abs_input, sympt, '**', "*ecto*.*"),
                                            recursive=True))
        # Get verso image
        lt_verso = sorted(glob.glob(op.join(abs_input, sympt, '**', "*erso*.*"),
                                            recursive=True))
        # Verify same number of images
        if len(lt_recto)!=len(lt_verso):
            raise Exception('Number of recto and verso different')

        # Get number of recto image
        nb_recto = len(lt_recto)

        # Verify stay enough image for split
        if np.around((nb_recto*(pp_stack)) / args.split) < 1:
            raise Exception('Zero image for test of stacking for ' + sympt)

        all_img[RECTO] += lt_recto
        all_img[VERSO] += lt_verso
        all_img[LAB] += [sympt] * nb_recto

    # Get absolu path of output recto directory
    abs_recto = op.abspath(op.expanduser(args.dir_rec))
    # Get absolu path of output verso directory
    abs_verso = op.abspath(op.expanduser(args.dir_ver))

    # Create directory struct
    create_struct_dir(abs_recto, args.split)
    create_struct_dir(abs_verso, args.split)

    # Define Train and valid
    spl = ms.StratifiedShuffleSplit(1, train_size=args.train,
                                    test_size=args.val)
    train_val = list(spl.split(all_img[RECTO], all_img[LAB]))[0]

    for index, part in enumerate([TRAIN, VAL]):
        fit_directory(op.join(abs_recto, CST_DL, part), RECTO, all_img,
                      train_val[index])
        fit_directory(op.join(abs_verso, CST_DL, part), VERSO, all_img,
                      train_val[index])

    # Define stacking
    # Keep all index not use in train and validation
    dl_index = np.concatenate((train_val[0], train_val[1]))
    stack_index = set(list(range(len(all_img[RECTO])))).difference(dl_index)
    stack_index = list(stack_index)
    stack_sympt = np.array(all_img[LAB])[stack_index]

    # Define k-fold for stacking
    spl = ms.StratifiedKFold(args.split)
    for split, train_test in enumerate(spl.split(stack_index, stack_sympt)):
        for index, part in enumerate([TRAIN, TEST]):
            fit_directory(op.join(abs_recto, CST_STACK, str(split), part),
                          RECTO, all_img,
                          np.array(stack_index)[train_test[index]])
            fit_directory(op.join(abs_verso, CST_STACK, str(split), part),
                          VERSO, all_img,
                          np.array(stack_index)[train_test[index]])


if __name__=='__main__':
    run()
