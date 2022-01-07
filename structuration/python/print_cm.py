#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 10:20:10 2022

@author: port-mpalerme

manage confusion matrix
"""
import csv
import sklearn.metrics as sm
import numpy as np

__all__ = ['fill_csv_cm', 'fill_cvs_stat', 'print_csv']

def fill_csv_cm(in_pred, in_true, in_lab=None):
    """
    Create rows of confusion matrix for .csv file

    Parameters
    ----------
    in_pred : list
        list of prediction.
    in_true : list
        list of true.
    in_lab : list, optional
        list of labels. The default is None.

    Returns
    -------
    tab : list of list
        rows of confusion matrix for .csv file.

    """
    tab = []
    infos = [["Confusion matrix not normalized", None],
             ["Confusion matrix normalized", 'true']]
    # Get labels
    if in_lab is None:
        v_lab = np.unique(in_true)
    else:
        v_lab = in_lab

    for info in infos:
        # Add title of confusion matrix
        tab.append([info[0]])
        # Generate confusion matrix
        confmat = sm.confusion_matrix(in_true, in_pred, labels=v_lab,
                                      normalize=info[1])
        # Add header of confusion matrix
        tab.append(["", ""] + list(v_lab))
        # Add row of confusion matrix
        for index, row in enumerate(confmat):
            tab.append(["", v_lab[index]] + list(row))
        tab.append([])
    return tab

def fill_cvs_stat(in_pred, in_true, in_lab=None):
    """
    create rows of confusion matrix stats for .csv file

    Parameters
    ----------
    in_pred : list
        list of prediction.
    in_true : list
        list of true.
    in_lab : list, optional
        list of labels. The default is None.

    Returns
    -------
    tab : list of list
        rows of confusion matrix stats for .csv file.

    """
    tab = []

    # Get labels
    if in_lab is None:
        v_lab = np.unique(in_true)
    else:
        v_lab = in_lab

    dic_stat = sm.classification_report(in_true, in_pred, labels=v_lab,
                                        output_dict=True)

    # Add accuracy
    tab.append(['accuracy', dic_stat['accuracy']])
    tab.append([])
    del dic_stat['accuracy']

    # Add header of stats
    tab.append([""] + list(dic_stat[str(v_lab[0])]))

    # Add each row of stats
    for key in dic_stat:
        tab.append([key])
        for key_stat in dic_stat[key]:
            tab[-1].append(dic_stat[key][key_stat])
    tab.append([])

    return tab

def cm_n_stat_csv(in_pred, in_true, in_lab=None):
    """
    create rows of confusion matrix and stats for .csv file

    Parameters
    ----------
    in_pred : list
        list of prediction.
    in_true : list
        list of true.
    in_lab : list, optional
        list of labels. The default is None.

    Returns
    -------
    rows : list of list
        rows of confusion matrix and stats for .csv file.

    """
    rows = fill_csv_cm(in_pred, in_true, in_lab)
    rows += fill_cvs_stat(in_pred, in_true, in_lab)

    return rows

def cross_val_csv(in_preds, in_trues, in_lab=None, in_iter_name=None):
    """
    Create rows of confusion matrix and stats for csv file.
    One global and one by iteration

    Parameters
    ----------
    in_preds : list of list
        prediction of each iteration.
    in_trues : list of list
        trues of each iteration.
    in_lab : list, optional
        list of labels. The default is None.
    in_iter_name : {None, str, list}, optional
        if str is prefix. If list . The default is None.

    Raises
    ------
    Exception
        number of name iteration different of iteration.

    Returns
    -------
    rows : list of list
        rows of confusion matrix and stats for .csv file..

    """
    # Calculate global confusion matrix and stat of cross validation
    rows = [["Result global"]]
    rows += cm_n_stat_csv(np.concatenate(in_preds), np.concatenate(in_trues),
                          in_lab)

    if isinstance(in_iter_name, str):
        v_iter_name = [in_iter_name + '_' + index
                       for index in range(len(in_preds))]
    elif isinstance(in_iter_name, list):
        # Verify number of name iteration same of iteration
        if len(in_iter_name) == len(in_preds):
            v_iter_name = in_iter_name
        else:
            raise Exception("number of name iteration different of iteration")
    elif in_iter_name is None:
        v_iter_name = ['Iteration_' + str(index)
                       for index in range(len(in_preds))]

    # Calculate confusion matrix and stat of each iteration
    for pred, true, name in zip(in_preds, in_trues, v_iter_name):
        rows.append([name])
        rows += cm_n_stat_csv(pred, true, in_lab)

    return rows


def print_csv(rows, out_file):
    """
    save all rows in .csv file

    Parameters
    ----------
    rows : list of list
        All csv's rows .
    out_file : str
        path of .csv file.

    Returns
    -------
    None.

    """
    with open(out_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
