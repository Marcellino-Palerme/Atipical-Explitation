#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 12:20:24 2021

@author: mpalerme

cm_print module print confusion matirx on csv file
"""

import os.path as op
import csv
import tools_file as tf
import sklearn.metrics as slm

__all__ = ['cm_print']


def cm_print(fname, y_true, y_pred, labels=None, normalize=False):
    """
    Calculate the confusion matrix and print it in csv file

    Parameters
    ----------
    fname : str
        Target Filename.
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    labels : array-like of shape (n_classes), optional
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If None is given, those that appear at
        least once in y_true or y_pred are used in sorted order.
        The default is None.
    normalize : bool, optional
        Normalizes confusion matrix over all the population. If False confusion
        matrix will no be normalized. The default is False.

    Returns
    -------
    None.

    """
    if normalize :
        normalize = 'true'
    else:
        normalize = None

    # Calculate the confusion matrix
    my_cm = slm.confusion_matrix(y_true,
                                 y_pred,
                                 labels=labels,
                                 normalize=normalize)

    # Extract directory of file's path
    path = op.dirname(fname)
    tf.create_directory(path)

    # Create csv
    f_csv = open(fname, 'w')

    # Create write for csv
    writer = csv.writer(f_csv)

    # Save raw confusion matrix
    if labels is None:
        writer.writerows(my_cm)
        f_csv.close()
        return 0

    # Write header
    header = list(labels)
    header.insert(0, "classes")
    writer.writerow(header)

    # Write each line
    for index, label in enumerate(labels):
        line = list(my_cm[index])
        line.insert(0, label)
        writer.writerow(line)

    f_csv.close()
    return 0
