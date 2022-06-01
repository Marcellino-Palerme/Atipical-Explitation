#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 12:13:51 2022

@author: Marcellino Palerme

Provide functions to prepare data for deep Learning
"""
from glob import glob
import os.path as op
import PIL.Image as im
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit as sss
from skimage import io
import skimage.transform as tfm
import h5py
import tools_file as tsf
import random
###############################################################################
### functions
###############################################################################
def resize_image(im_in, cols, rows, preserve_ration=True):
    """
    Modify size of image

    Parameters
    ----------
    im_in : PIL.Image
        images.
    cols : int
        Number of columns of new images.
    rows : int
        Number of rows of new images.

    preserve_ration : bool, optional
        Do it preserve ratio cols*rows of originial image. The default is True.

    Returns
    -------
    PIL.Image: resized image

    """
    # Resize direct if we don't preserve ratio
    if not preserve_ration:
        return im_in.resize((cols, rows))

    # Create a black image with new size
    im_backg = im.new('RGB', (cols,rows))

    # Search the highter ratio
    ratio = max(im_in.width/cols, im_in.height/rows)

    # We resize image
    im_tmp = im_in.resize((int(im_in.width/ratio), int(im_in.height/ratio)))

    # Calculate position of image to centered it
    pos = (int(abs(im_tmp.width-cols)/2), int(abs(im_tmp.height-rows)/2))

    # We put resized image on black image
    im_backg.paste(im_tmp, pos)

    return im_backg

def resize_images(dir_in, cols, rows, dir_out, preserve_ration=True):
    """
    Modify size of images

    Parameters
    ----------
    dir_in : str
        Directory where are images.
    cols : int
        Number of columns of new images.
    rows : int
        Number of rows of new images.
    dir_out : str
        Directory where save new images.
    preserve_ration : bool, optional
        Do it preserve ratio cols*rows of originial image. The default is True.

    Returns
    -------
    None.

    """
    # Take absolute path of input directory
    abs_dir_in = op.abspath(op.expanduser(dir_in))

    # Take absolute path of output directory
    abs_dir_out = op.abspath(op.expanduser(dir_out))

    tsf.create_directory(abs_dir_out)

    # Take list of all files or directory
    lt_files = glob(op.join(abs_dir_in, '*.*'))

    for my_file in lt_files:
        name_file = op.basename(my_file)
        try:
            my_im = im.open(my_file)
            my_im = resize_image(my_im, cols, rows, preserve_ration)
            my_im.save(op.join(abs_dir_out, name_file))
        except IOError:
            pass

def split(splits, data):
    """


    Parameters
    ----------
    splits : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Verify if sum of split equal one
    if np.sum(splits) != 1:
        raise ValueError("Sum of split is different of one")

    # Copy data
    tmp = data.iloc[:, :]
    res_split = []

    for index, my_split in enumerate(splits[:-1]):
        ratio_train = my_split/np.sum(np.array(splits)[index :])
        v_sss = sss(1, train_size=ratio_train)
        train_ind, test_ind = next(v_sss.split(tmp.iloc[:, 0],
                                               tmp.iloc[:, 1]), None)
        res_split.append(data.iloc[train_ind])
        tmp = data.iloc[test_ind]

    res_split.append(tmp)

    return res_split

def save_cluster(path, l_img_r, l_img_v, l_lab, l_aug):
    """


    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    l_img_r : TYPE
        DESCRIPTION.
    l_img_v : TYPE
        DESCRIPTION.
    l_lab : TYPE
        DESCRIPTION.
    l_aug : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    h5f = h5py.File(path, 'w')

    h5f.create_dataset('image_r', data=l_img_r)
    h5f.create_dataset('image_v', data=l_img_v)
    h5f.create_dataset('label', data=l_lab)
    h5f.create_dataset('augmented', data=l_aug)

    h5f.close()

def aug_image(image, ratio_stay):
    """


    Parameters
    ----------
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    image

    """
    nb_pixel = np.nonzero(image)[0].shape[0]
    tmp_ratio = 0

    while(tmp_ratio < ratio_stay):
        tform = tfm.SimilarityTransform(scale=random.uniform(0.7,1.3),
                                        rotation=random.uniform(-np.pi, np.pi))
        tmp_img = tfm.warp(image, tform.inverse)
        tmp_nb_pixel = np.nonzero(tmp_img)[0].shape[0]
        tmp_ratio = tmp_nb_pixel / nb_pixel
    tmp_img *= 255
    return tmp_img.astype(np.uint8)


def create_cluster(dir_in, name, clus_size, nb_aug, splits=(0.8, 0.2),
                   dir_out=None):
    """
    Create cluster

    Parameters
    ----------
    dir_in : str
        Directory where each label type image in a directory.
    name : str
        Prefixe name of cluster.
    clus_size : int
        Number of images by cluster.
    nb_aug : int
        Number of augmentation.
    dir_out : str, optional
        Directory where save clusters. The default is None.

    Returns
    -------
    None.

    """
    # Take absolue path of input directory
    abs_dir_in = op.abspath(op.expanduser(dir_in))

    # Get all elements in input directory
    all_elemt = glob(op.join(abs_dir_in, '*'))

    # Get all directories of input directory
    all_dir = [x for x in all_elemt if op.isdir(x)]

    df_im_lab = pd.DataFrame(columns=['image', 'label', 'augmented'])

    # Get all images name with label
    for one_dir in all_dir:
        # Extract label
        label = op.basename(one_dir)
        # Get all file names
        all_files = glob(op.join(one_dir, 'recto', '*.*'))

        # Create dictionary
        tmp = {'image': all_files, 'label': [label] * len(all_files),
               'augmented': [False] * len(all_files)}
        tmp_aug ={'image': all_files * nb_aug,
                  'label': [label] * len(all_files) * nb_aug,
                  'augmented': [True] * len(all_files) * nb_aug}

        # Transform dictionary in dataframe
        tmp = pd.DataFrame(tmp)
        tmp_aug = pd.DataFrame(tmp_aug)

        # Add in dataframe
        df_im_lab = pd.concat([df_im_lab, tmp, tmp_aug], ignore_index=True)

    # Divide data
    a_splits = split(splits, df_im_lab)

    if dir_out is None:
        abs_dir_out = abs_dir_in
    else:
        # Take absolue path of input directory
        abs_dir_out = op.abspath(op.expanduser(dir_out))

    # Create cluster
    for index, section in enumerate(a_splits):
        tsf.create_directory(op.join(abs_dir_out,f'sect{index:d}'))
        l_img_r = []
        l_img_v = []
        l_lab = []
        l_aug = []
        num_clus = 0
        for _, row in section.iterrows():
            l_img_r.append(io.imread(row.image))
            row.image = row.image.replace('recto', 'verso')
            row.image = row.image.replace('Recto', 'Verso')
            l_img_v.append(io.imread(row.image))
            l_lab.append(row.label)
            l_aug.append(row.augmented)

            # augmented image
            if l_aug[-1]:
                l_img_r[-1] = aug_image(l_img_r[-1], 0.7)
                l_img_v[-1] = aug_image(l_img_v[-1], 0.7)

            # Cluster is full
            if len(l_lab)==clus_size:
                save_cluster(op.join(abs_dir_out,f'sect{index:d}',
                                     f'{name}_sect{index:d}_part{num_clus:03d}.h5'),
                             l_img_r, l_img_v, l_lab, l_aug)

                l_img_r = []
                l_img_v = []
                l_lab = []
                l_aug = []
                num_clus += 1

        # if cluster not complet
        save_cluster(op.join(abs_dir_out,f'sect{index:d}',
                             f'{name}_sect{index:d}_part{num_clus:03d}.h5'),
                     l_img_r, l_img_v, l_lab, l_aug)




###############################################################################
### Main function
###############################################################################
# def run():
#     """
#     main function

#     Returns
#     -------
#     None.

#     """

# if __name__=='__main__':
#     run()
