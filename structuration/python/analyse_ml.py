#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 07:57:47 2022

@author: port-mpalerme

Generate analyse from execution of machine learning
"""
import argparse
import os.path as op
import numpy as np
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
    parser.add_argument('-i', '--in', type=str,
                        help="input path file",
                        required=True, dest='fin')

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
    abs_fin = op.abspath(op.expanduser(args.fin))
    abs_fout = op.abspath(op.expanduser(args.fout))

    with open(abs_fin, 'r') as fjson:
        djson = json.load(fjson)

    rows = []
    for key in djson:
        if key == 'symptoms':
            continue
        rows.append([key])
        preds = [np.array(djson['symptoms'])[index]
                 for index in djson[key][0]['predict']]
        trues = [np.array(djson['symptoms'])[index]
                 for index in djson[key][0]['true']]
        rows += pm.cross_val_csv(preds, trues, djson['symptoms'])

    pm.print_csv(rows, abs_fout)

if __name__=='__main__':
    run()
