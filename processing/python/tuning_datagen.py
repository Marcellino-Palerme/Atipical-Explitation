#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:29:13 2022

@author: Marcellino Palerme


"""
from glob import glob
import os.path as op
import random as rd
import h5py
import numpy as np
from tensorflow.keras.utils import Sequence



class DataGenerator(Sequence):
    """
    Datagenerator
    """

    def __init__(self, clusters_dir, batch_size=32, recto=True,
                 verso=False):
        """


        Parameters
        ----------
        clusters_dir : TYPE
            DESCRIPTION
        batch_size : TYPE, optional
            DESCRIPTION. The default is 32.
        recto : TYPE, optional
            DESCRIPTION. The default is True.
        verso : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        # Take absolue path of cluster directory
        self.abs_clus_dir = op.abspath(op.expanduser(clusters_dir))
        # Take list of cluster file
        self.l_cluster = glob(op.join(clusters_dir, '*.h5'))

        self.batch_size = batch_size
        self.recto = recto
        self.verso = verso
        self.sympt = {'Alt':0, 'Big':1, 'Mac':2, 'Mil':3, 'Myc':4,
                      'Pse':5, 'Syl':6}


    def __len__(self):
        """


        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return len(self.l_cluster)

    def get_clus(self, index):
        """

        Parameters
        ----------
        index : TYPE
            DESCRIPTION.

        Returns
        -------
        dictionary.

        """
        h5f = h5py.File(self.l_cluster[index], 'r')

        dic = {'image_r': np.array(h5f['image_r'])}
        dic['image_v'] = np.array(h5f['image_v'])
        dic['label'] = np.array(h5f['label']).astype(str)
        dic['label'] = np.array([self.sympt[x] for x in dic['label']])

        h5f.close()

        return dic

    def increase_clus(self, main_clus, index, size):
        """


        Parameters
        ----------
        main_clus : TYPE
            DESCRIPTION.
        index : TYPE
            DESCRIPTION.
        size : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        h5f = h5py.File(self.l_cluster[index], 'r')

        dic = {'image_r': np.array(h5f['image_r'])[:size]}
        dic['image_v'] = np.array(h5f['image_v'])[:size]
        dic['label'] = np.array(h5f['label'])[:size].astype(str)
        dic['label'] = np.array([self.sympt[x] for x in dic['label']])

        h5f.close()

        return {key:np.concatenate((main_clus[key], dic[key]), axis=0)
                for key in dic}

    def __getitem__(self, index):
        """


        Parameters
        ----------
        index : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        clus = self.get_clus(index)
        # Verify cluster containt
        if len(clus['label']) < self.batch_size:
            if index == 0:
                clus = self.increase_clus(clus, 1, self.batch_size -
                                                   len(clus['label']))
            else:
                clus = self.increase_clus(clus, 0, self.batch_size -
                                                   len(clus['label']))

        if self.recto & self.verso:
            return [clus['image_r'], clus['image_v']], clus['label']

        if self.recto:
            return clus['image_r'], clus['label']

        return clus['image_v'], clus['label']

    def on_epoch_end(self):
        """


        Returns
        -------
        None.

        """
        # shuffle order to read cluster
        rd.shuffle(self.l_cluster)
