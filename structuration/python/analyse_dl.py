#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 14:22:34 2022

@author: port-mpalerme

Generate analyse from execution of deep learning
"""
import argparse
import os.path as op
import glob
import json
import print_cm as pm

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
    parser.add_argument('-i', '--dir_in', type=str,
                        help="input path directory",
                        required=True, dest='din')

    # Add argument for source directory
    parser.add_argument('-o', '--out', type=str,
                        help="output path file",
                        required=True, dest='fout')

    # Take all arguments
    return parser.parse_args()

def run():
    """
    main function

    Returns
    -------
    None.

    """
    # Take input arguments
    args = arguments()

    # Get absolu path
    abs_din = op.abspath(op.expanduser(args.din))
    abs_fout = op.abspath(op.expanduser(args.fout))

    # Take all file result
    lt_files = sorted(glob.glob(op.join(abs_din, '*pred_true*')))

    rows = []

    for f_result in lt_files:
        # Create title of part of csv
        # Keep name of file
        title = op.basename(f_result)
        # Delete extension
        title = op.splitext(title)[0]
        # Keep model used
        title = title[26:]

        rows.append([title])

        with open(f_result, 'r') as fjson:
            djson = json.load(fjson)

        rows += pm.cm_n_stat_csv(djson['test_pred'], djson['test_true'])

    pm.print_csv(rows, abs_fout)

if __name__=='__main__':
    run()
