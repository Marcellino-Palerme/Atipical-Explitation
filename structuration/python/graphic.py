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
    Create bar graphic to compare recto and rectoverso in machine learning

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
    Create bar graphic to compare recto and rectoverso in deep learning

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
    lt_r = [float(dic_acc_r[key]["global"]) for key in lt_impl_r]


    dic_temp = {'impl':lt_impl_r, 'recto': lt_r}

    for verso in lt_impl_r:
        dic_temp[verso] = []
        for recto in lt_impl_r:
            r_v = recto + '_' + verso
            # Take all global acc of recto verso
            dic_temp[verso].append(float(dic_acc_rv[r_v]["global"]))

    # Create dataframe
    df_acc = pd.DataFrame(dic_temp)
    # Create plotbar from dataframe
    ax_bar = df_acc.set_index('impl').plot.bar(ylim=(0,1), width=0.9)
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

def graph_box_ml(dic_acc_r, dic_acc_rv, title_r, title_rv, out_file):
    """
    Create box graphic to compare recto and rectoverso in machine learning

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
    # Create dataframe
    df_acc = pd.DataFrame()

    # Take all implementation
    for impl in sorted(dic_acc_r.keys()):
        # Take all iteration acc of recto
        df_acc[impl + '_recto'] = np.array(dic_acc_r[impl]["iter"]).\
                                          astype(float)
        # Take all iteration acc of recto verso
        df_acc[impl + '_rectoverso'] = np.array(dic_acc_rv[impl]["iter"]).\
                                               astype(float)

    # Create plotbar from dataframe
    df_acc.plot.box()
    # Put vertical x values
    plt.xticks(rotation=90)
    # Delete x label
    plt.xlabel('')
    # Auto-adjust
    plt.tight_layout()
    # Save plot in file
    plt.savefig(out_file)

def graph_box_dl(dic_acc_r, dic_acc_rv, out_file):
    """
    Create box graphic to compare recto and rectoverso in deep learning

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

    for index, recto in enumerate(lt_impl_r):
        # Create dataframe
        df_acc = pd.DataFrame()
        for verso in lt_impl_r:
            r_v = recto + '_' + verso
            # Take all iteration acc of recto verso
            df_acc[r_v] = np.array(dic_acc_rv[r_v]['iter']).astype(float)
        # Plot DataFrame
        df_acc.plot.box()
        # Add line of recto
        plt.axhline(float(dic_acc_r[recto]["global"]), color='red',
                    ls='dotted')
        # Auto-adjust
        plt.tight_layout()

        # Save plot in file
        plt.savefig(out_file.replace('.', '_' + recto + '.'))

def graph_diff_acc_ml(dic_acc_r, dic_acc_rv, out_file):
    """
    Create line graphic to show diff between global acc and symptom acc
    in machine learning

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
    lt_impl_rv = [impl + '_rv' for impl in lt_impl_r]
    # Take list of symptom
    lt_sympt = [sympt for sympt in sorted(dic_acc_r[lt_impl_r[0]].keys())
                      if len(sympt)==3]

    dc_acc = {'impl':sorted(lt_impl_r + lt_impl_rv)}
    for sympt in lt_sympt:
        dc_acc[sympt]=[]
        for impl in lt_impl_r:
            dc_acc[sympt].append(float(dic_acc_r[impl][sympt]["global"]) -
                                 float(dic_acc_r[impl]["global"]))
            dc_acc[sympt].append(float(dic_acc_rv[impl][sympt]["global"]) -
                                 float(dic_acc_rv[impl]["global"]))

    # Create dataframe
    df_acc = pd.DataFrame(dc_acc)
    # Create plotbar from dataframe
    df_acc.set_index('impl').plot.line(xlabel=dc_acc['impl'])
    # Put vertical x values
    plt.xticks(rotation=90)
    # Delete x label
    plt.xlabel('')
    # Put legend out of plot
    plt.legend(bbox_to_anchor = (1.05, 0.6))
    # Auto-adjust
    plt.tight_layout()
    # Save plot in file
    plt.savefig(out_file)

def graph_diff_acc_dl(dic_acc_r, dic_acc_rv, out_file):
    """
    Create line graphic to show diff between global acc and symptom acc
    in deep learning

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
    dic_fusion = dic_acc_r.copy()
    dic_fusion.update(dic_acc_rv)

    dc_acc = {'impl':sorted(dic_fusion.keys())}

    # Take list of symptom
    lt_sympt = [sympt for sympt in sorted(dic_acc_r[dc_acc['impl'][0]].keys())
                      if len(sympt)==3]


    for sympt in lt_sympt:
        dc_acc[sympt]=[]
        for impl in dc_acc['impl']:
            dc_acc[sympt].append(float(dic_fusion[impl][sympt]["global"]) -
                                 float(dic_fusion[impl]["global"]))

    # Create dataframe
    df_acc = pd.DataFrame(dc_acc)
    # Create plotbar from dataframe
    df_acc.set_index('impl').plot.line()
    # Put vertical x values
    plt.xticks(rotation=90, size='small')
    # Delete x label
    plt.xlabel('')
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
                     'ML rectoverso', op.join(abs_dout, 'Compare_Acc_ML.png'))

    graph_box_ml(dic_acc['mlr'], dic_acc['mlrv'], 'ML recto', 'ML rectoverso',
                 op.join(abs_dout, 'Box_Acc_ML.png'))

    graph_diff_acc_ml(dic_acc['mlr'], dic_acc['mlrv'],
                      op.join(abs_dout, 'Diff_Acc_ML.png'))

    graph_compare_dl(dic_acc['dlr'], dic_acc['dlrv'],
                     op.join(abs_dout, 'Compare_Acc_DL_mv.png'))

    graph_diff_acc_dl(dic_acc['dlr'], dic_acc['dlrv'],
                      op.join(abs_dout, 'Diff_Acc_DL_mv.png'))

    graph_compare_dl(dic_acc['dlr'], dic_acc['stack'],
                     op.join(abs_dout, 'Compare_Acc_DL_stack.png'))

    graph_box_dl(dic_acc['dlr'], dic_acc['stack'],
                 op.join(abs_dout, 'Box_Acc_DL_stack.png'))

    graph_diff_acc_dl(dic_acc['dlr'], dic_acc['stack'],
                      op.join(abs_dout, 'Diff_Acc_DL_stack.png'))

if __name__ == "__main__":
    run()
