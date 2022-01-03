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
from sklearn.model_selection import StratifiedKFold
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
CST_TRAIN = "train"
CST_VAL = "validation"


##############################################################################
### Additional function
##############################################################################
def create_directory_split(path, nb_split, rectoverso):
    """
    Create all directory save kfold split

    Parameters
    ----------
    path : str
        Path of directory where save splits.
    nb_split : int
        Number of split.
    rectoverso : bool
        work with verso.
    Returns
    -------
    None.

    """

    # Take absolue path
    abs_path = op.abspath(path)

    # Create all split directory
    for index in range(nb_split):
        for symptom in LT_CLASS:
            tf.create_directory(op.join(abs_path, str(index), TRAIN, symptom,
                                        RECTO))
            tf.create_directory(op.join(abs_path, str(index), TEST, symptom,
                                        RECTO))
            if rectoverso:
                tf.create_directory(op.join(abs_path, str(index), TRAIN,
                                            symptom, VERSO))
                tf.create_directory(op.join(abs_path, str(index), TEST,
                                            symptom, VERSO))

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
    parser.add_argument('-v', '--validation',
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

    # Get absolu path of input directory
    abs_input = op.abspath(op.expanduser(args.dir_in))

    # Get number of recto image
    nb_recto = len(glob.glob(op.join(abs_input, '*', '**', "*ecto*.*"),
                             recursive=True))

    # Verify stay enough image for split
    if np.around((nb_recto*(1-args.train-args.val)) / args.split) < 1:
        raise Exception('Zero image for test of stacking')

    # TODO : implemented Train/Validation/Split


    abs_output = op.abspath(op.expanduser(args.dir_out))

    # Create output directory
    create_directory_split(abs_output, args.nb_split, args.rv)


    ###########################################################################
    ### Create dataset
    ### Dataset contain name of each image and number of symptom of image
    ###########################################################################

    d_dataset = {RECTO :[], "symptom" : []}

    # Work with verso too
    if args.rv:
        d_dataset[VERSO] = []

    for index, symptom in enumerate(LT_CLASS):
        # Take all recto image name
        lt_name = sorted(glob.glob(op.join(abs_input, symptom, '**',
                                           "*ecto*.*"),
                                   recursive=True))

        # Add recto to dataset
        d_dataset[RECTO] = d_dataset[RECTO] + lt_name

        # Work with verso
        if args.rv:
            # Take all verso image name
            lt_name = sorted(glob.glob(op.join(abs_input, symptom, '**',
                                               "*erso*.*"),
                                       recursive=True))

            # Add verso to dataset
            d_dataset[VERSO] = d_dataset[VERSO] + lt_name

        # indicate the symptom index
        d_dataset["symptom"] = d_dataset["symptom"] +\
                               list(np.ones(len(lt_name), np.int8) * index)


    ###########################################################################
    ### Split dataset with Kfold
    ###########################################################################

    # Define number of split
    skf = StratifiedKFold(n_splits=args.nb_split, shuffle=True)

    for index_split, train_test_index in enumerate(skf.split(d_dataset[RECTO],
                                                             d_dataset['symptom'])
                                                   ):
        index_split = str(index_split)

        for index in train_test_index[0]:
            # Take symptom name
            symptom = LT_CLASS[d_dataset["symptom"][index]]
            # Take image name
            image_name = op.basename(d_dataset[RECTO][index])
            # Create symbolic link in train part of output directory
            symlink(d_dataset[RECTO][index], op.join(abs_output,index_split,
                                                     TRAIN, symptom, RECTO,
                                                     image_name))
            # if work with verso
            if args.rv:
                # Take image name
                image_name = op.basename(d_dataset[VERSO][index])
                # Create symbolic link in train part of output directory
                symlink(d_dataset[VERSO][index], op.join(abs_output,
                                                         index_split,
                                                         TRAIN, symptom,
                                                         VERSO, image_name))

        for index in train_test_index[1]:
            # Take symptom name
            symptom = LT_CLASS[d_dataset["symptom"][index]]
            # Take image name
            image_name = op.basename(d_dataset[RECTO][index])
            # Create symbolic link in train part of output directory
            symlink(d_dataset[RECTO][index], op.join(abs_output,index_split,
                                                     TEST, symptom, RECTO,
                                                     image_name))
            # if work with verso
            if args.rv:
                # Take image name
                image_name = op.basename(d_dataset[VERSO][index])
                # Create symbolic link in train part of output directory
                symlink(d_dataset[VERSO][index], op.join(abs_output,
                                                         index_split,
                                                         TEST, symptom, VERSO,
                                                         image_name))

if __name__=='__main__':
    run()
