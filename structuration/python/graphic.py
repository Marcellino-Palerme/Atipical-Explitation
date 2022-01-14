#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 07:21:56 2022

@author: port-mpalerme

Create graphics to analyse
"""

import argparse
import os.path as op
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    # Add argument for source path
    parser.add_argument('-i', '--in', type=str,
                        help="input path file",
                        required=True, dest='fin')

    # Add argument for output directory
    parser.add_argument('-o', '--out', type=str,
                        help="output path directory",
                        required=True, dest='dout')

    # Take all arguments
    return parser.parse_args()

###############################################################################
### additionnal function
###############################################################################
def bar_distrubution(nb_bar, ax_bar):
    """
    Calculate bar distribution for one x

    Parameters
    ----------
    nb_bar : int
        number of bar by x.
    ax_bar : matplotlib.axes
        axes of plotbar.

    Returns
    -------
    list
        Distrubition where start each bar for one x.

    """
    # Calculate bar mean size
    bar_size = np.mean([info.get_width() for info in ax_bar.patches])

    if nb_bar%2 == 0:
        return np.array(range(int(-nb_bar/2), nb_bar)) * bar_size

    temp = int((nb_bar-1)/2)
    return np.array(range(-temp, temp + 1)) * bar_size - bar_size/2

def write_value(lt_yr, lt_yrv, ax_bar):
    """
    Add value to top of each bar

    Parameters
    ----------
    lt_yr : list
        All values of recto.
    lt_yrv : list or list of list
        All values of recto-verso.
    ax_bar : matplotlib axes
        axes of plotbar.

    Returns
    -------
    None.

    """
    # Concatenate all y
    lt_y = np.concatenate((np.array(lt_yr).reshape((-1,1)),
                           np.array(lt_yrv).reshape((len(lt_yr),-1))), axis=1)
    # Get position of bar group
    xlocs = plt.xticks()

    # Calculate distribution of bar group
    bar_dist = bar_distrubution(len(lt_y[0]), ax_bar)

    # Add value to top of each bar
    for index,  xloc in enumerate(xlocs[0]):
        temp_dist = bar_dist + xloc
        for value, pos_x in zip(lt_y[index], temp_dist):
            plt.text(pos_x, value + 0.02, value, rotation="vertical")

def graph_compare_ml(dic_acc_r, dic_acc_rv, title_r, title_rv, out_file):
    """
    Create graphic to compare recto and rectoverso in machine learning

    Parameters
    ----------
    dic_acc_r : TYPE
        DESCRIPTION.
    dic_acc_rv : TYPE
        DESCRIPTION.
    title_r : TYPE
        DESCRIPTION.
    title_rv : TYPE
        DESCRIPTION.
    out_file : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Take all implementation
    lt_impl_r = sorted(dic_acc_r.keys())
    # Take all global acc of recto
    lt_r = [float(dic_acc_r[key]["global"]) for key in lt_impl_r]
    # Take all global acc of recto verso
    lt_rv = [float(dic_acc_rv[key]["global"]) for key in lt_impl_r]

    # Create dataframe
    df_acc = pd.DataFrame({'impl':lt_impl_r, title_r: lt_r, title_rv:lt_rv})
    # Create plotbar from dataframe
    ax_bar = df_acc.set_index('impl').plot.bar(ylim=(0,1))
    # Delete x label
    plt.xlabel('')
    # Add value on top of each bar
    write_value(lt_r, lt_rv, ax_bar)
    # Put legend out of plot
    plt.legend(bbox_to_anchor = (1.05, 0.6))
    # Auto-adjust
    plt.tight_layout()
    # Save plot in file
    plt.savefig(out_file)


def graph_compare_dl(dic_acc_r, dic_acc_rv, out_file):
    """
    Create graphic to compare recto and rectoverso in deep learning

    Parameters
    ----------
    dic_acc_r : TYPE
        DESCRIPTION.
    dic_acc_rv : TYPE
        DESCRIPTION.
    out_file : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Take all implementation
    lt_impl_r = sorted(dic_acc_r.keys())
    # Take all global acc of recto
    lt_r = [float(dic_acc_r[key]["iter"][0]) for key in lt_impl_r]


    dic_temp = {'impl':lt_impl_r, 'recto': lt_r}

    # Verify if there is a global Acc
    # Case where there is on iteration
    if dic_acc_rv[lt_impl_r[0] + '_' + lt_impl_r[0]]["global"] is None:
        glob_iter = '["iter"][0]'
    else:
        glob_iter = '["global"]'

    for verso in lt_impl_r:
        dic_temp[verso] = []
        for recto in lt_impl_r:
            r_v = recto + '_' + verso
            # Take all global acc of recto verso
            temp = eval("dic_acc_rv[r_v]" + glob_iter)
            dic_temp[verso].append(float(temp))

    # Create dataframe
    df_acc = pd.DataFrame(dic_temp)
    # Create plotbar from dataframe
    ax_bar = df_acc.set_index('impl').plot.bar(ylim=(0,1))
    # Delete x label
    plt.xlabel('')
    lt_rv =[dic_temp[key] for key in dic_temp if key not in ('impl', 'recto')]
    # Add value on top of each bar
    write_value(lt_r, np.array(lt_rv).T, ax_bar)
    # Put legend out of plot
    plt.legend(bbox_to_anchor = (1.05, 0.6))
    # Auto-adjust
    plt.tight_layout()
    # Save plot in file
    plt.savefig(out_file)

###############################################################################
### main function
###############################################################################
def run():
    """


    Returns
    -------
    None.

    """
    # take all input arguments
    args = arguments()
    # Take absolue path of file
    abs_fin = op.abspath(op.expanduser(args.fin))
    # Take absolue path of output directory
    abs_dout = op.abspath(op.expanduser(args.dout))

    # Get all Acc
    with open(abs_fin, 'r') as fjson:
        dic_acc = json.load(fjson)

    graph_compare_ml(dic_acc['mlr'], dic_acc['mlrv'], 'ML recto',
                     'ML rectoverso', op.join(abs_dout, 'Compare_Acc_ML.jpg'))

    graph_compare_dl(dic_acc['dlr'], dic_acc['dlrv'],
                     op.join(abs_dout, 'Compare_Acc_DL_mv.jpg'))

    graph_compare_dl(dic_acc['dlr'], dic_acc['stack'],
                     op.join(abs_dout, 'Compare_Acc_DL_stack.jpg'))


if __name__ == "__main__":
    run()
