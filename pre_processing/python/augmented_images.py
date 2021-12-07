#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 08:37:36 2021

@author: port-mpalerme

Create augmented images
"""
import argparse
import os
import glob
import tensorflow as tf
import numpy as np

###############################################################################
### Function to verify argument input
###############################################################################
class NumberIterAction(argparse.Action):
    """
        manage the min of number of iteration
    """
    def __call__(self, parser, namespace, values, option_string=None):
        # Verify if value is strictly higher than 1
        if values <= 1:
            parser.error("Number of iteration have to be higher than 1")

        setattr(namespace, self.dest, values)

###############################################################################
### input arguments
###############################################################################
def arguments():
    """
    manage input arguments

    Returns
    -------
    namespace
        all input arguments.

    """
    parser = argparse.ArgumentParser()

    # Add argument for source directory
    parser.add_argument('-i', '--dir_in', type=str, help="source directory",
                        required=True, dest='dir_in')

    # Add argument to choose Recto or Recto/Verso
    parser.add_argument('--rv', action='store_true', help="Recto/Verso",
                        dest='rv')

    # Add argument indicate number of split
    parser.add_argument("-n", "--number_iter", type=int,
                        help="Number of iteration of augmentation",
                        default=4, dest='nb_iter', action=NumberIterAction)

    # Take all arguments
    return parser.parse_args()

###############################################################################
### additional functions
###############################################################################
def augmented_image(path_im, nb_iter, seq_aug):
    """
    create augmented images

    Parameters
    ----------
    path_im : str
        Path of image.
    nb_iter : int
        number of augmented images we create.
    seq_aug : Sequential
        Sequential of data augmentation

    Returns
    -------
    None.

    """
    # Get id of image
    id_im = os.path.basename(path_im)[0:21]
    # Get directory of image
    dir_im = os.path.dirname(path_im)
    # Get extension of image
    ext_im = os.path.splitext(path_im)[-1]

    # Take image
    img = tf.keras.preprocessing.image.load_img(path_im)
    # Transform image to array
    aimg = tf.keras.preprocessing.image.img_to_array(img)

    for index in range(nb_iter):
        img_name = "Au" + str(index) + "_" + id_im + ext_im
        aug_img = seq_aug(np.array([aimg]))
        tf.keras.preprocessing.image.save_img(os.path.join(dir_im, img_name),
                                              aug_img[0])

###############################################################################
### main function
###############################################################################
def run():
    """
    main function

    Returns
    -------
    None.

    """
    # Take input arguments
    args = arguments()
    # Take absolue path of input directory
    abs_dir_in = os.path.abspath(os.path.expanduser(args.dir_in))

    # Take all symptoms
    lt_symptoms = [sympt for sympt in os.listdir(abs_dir_in)\
                   if os.path.isdir(os.path.join(abs_dir_in,sympt))]

    # Create augmentation sequential
    exp = tf.keras.layers.experimental.preprocessing
    seq_aug = tf.keras.Sequential([exp.RandomRotation(0.5, 'constant'),
                                   exp.RandomZoom(0.2, 'constant')])

    for sympt in lt_symptoms:
        # Take recto images
        lt_img = sorted(glob.glob(os.path.join(abs_dir_in, sympt,'**',
                                                 '*ecto*.*'), recursive=True))

        # works with verso
        if args.rv:
            # Take verso images
            lt_verso = sorted(glob.glob(os.path.join(abs_dir_in, sympt, '**',
                                                     '*erso*.*'),
                                        recursive=True))
            # Add verso images
            lt_img += lt_verso


        for path_img in lt_img:
            augmented_image(path_img, args.nb_iter, seq_aug)

if __name__=='__main__':
    run()
