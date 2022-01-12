#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:38:53 2022

@author: mpalerme

generate json file with acc from all csv file
"""
import argparse
import os.path as op
import json
import csv


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

    # Add argument for ML recto csv file path
    parser.add_argument('-m', '--mlr', type=str,
                        help="input ML recto csv file path",
                        required=True, dest='mlr')

    # Add argument for ML recto-verso csv file path
    parser.add_argument('-l', '--mlrv', type=str,
                        help="input ML recto-verso csv file path",
                        required=True, dest='mlrv')

    # Add argument for DL recto csv file path
    parser.add_argument('-d', '--dlr', type=str,
                        help="input DL recto csv file path",
                        required=True, dest='dlr')

    # Add argument for DL recto-verso csv file path
    parser.add_argument('-v', '--dlrv', type=str,
                        help="input DL recto-verso csv file path",
                        required=True, dest='dlrv')

    # Add argument for Stacking recto-verso csv file path
    parser.add_argument('-s', '--stack', type=str,
                        help="input Stacking recto-verso csv file path",
                        required=True, dest='stack')

    # Add argument for source directory
    parser.add_argument('-o', '--out', type=str,
                        help="output path file",
                        required=True, dest='fout')

    # Take all arguments
    return parser.parse_args()

###############################################################################
### Additional function
###############################################################################
def extract_acc(in_path):
    """
    Take all acc in csv file

    Parameters
    ----------
    in_path : str
        csv file path.

    Returns
    -------
    dic : dictionary
        {implementation : {'global': Acc globlal,
                           'iter': list of Acc}}.

    """
    # Take absolue path of input
    in_path = op.abspath(op.expanduser(in_path))
    with open(in_path, 'r') as fcsv:
        reader = csv.reader(fcsv)
        dic = {}
        acc_global = False
        impl = ""
        for row in reader:
            # Identify in witch section we are
            if len(row)==1:
                if row[0] == "Result global":
                    acc_global = True
                elif not("Iteration_" in row[0] or "Confusion" in row[0]):
                    # Witch  implementation used
                    impl = row[0]
                    dic[impl] = {'iter':[], 'global':None}
            # Take Acc
            if len(row)==2:
                if acc_global:
                    dic[impl]['global'] = row[1]
                    acc_global = False
                else:
                    dic[impl]['iter'].append(row[1])

    return dic

###############################################################################
### Main function
###############################################################################
def run():
    """
    main function

    Returns
    -------
    None.

    """
    # Take all inputs arguments
    args = arguments()
    # Convert to dictionary
    dic = vars(args)
    # Take absolut path of output file
    abs_out = op.abspath(op.expanduser(dic['fout']))
    del dic['fout']

    extract = {}
    for key in dic:
        extract[key] = extract_acc(dic[key])

    with open(abs_out,'w') as fjson:
        json.dump(extract, fjson)


if __name__=='__main__':
    run()
