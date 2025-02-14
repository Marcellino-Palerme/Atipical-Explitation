#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:24:38 2022

@author: mpalerme

Convert a pickle file generated by haralick_n_moment_n_mlrlm_RV to json file
"""
import os.path as op
import argparse
import pickle
import json


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
    parser.add_argument('-i', '--file_in', type=str,
                        help="path of input serialized file",
                        required=True, dest='fin')

    # Add argument for output directory
    parser.add_argument('-o', '--file_out', type=str,
                        help="path of output json file", dest='fout')


    # Take all arguments
    return parser.parse_args()

##############################################################################
### Main function
##############################################################################
def run():
    """
    main function

    Returns
    -------
    None.

    """
    lt_sympt = ["Alt", "Big", "Mac", "Mil", "Myc", "Pse", "Syl"]

    args = arguments()
    # Get absolu path of input file
    fin = op.abspath(op.expanduser(args.fin))

    # Verify if output file define
    if args.fout is None:
        # By default output file has same name of input file with extension
        # json
        fout = fin + '.json'
    else:
        # Get absolu path of output file
        fout = op.abspath(op.expanduser(args.fout))

    # Get serialized dictionary
    with open(fin, 'rb') as my_file:
        dic = pickle.load(my_file)

    # Modify dictonary to be compatible with json format
    for key in dic:
        # delete classifier object
        del dic[key][0]['classifier']
        # Transform all ndarray to list
        for index, results in enumerate(dic[key][0]['predict']):
            results = results.astype(int)
            dic[key][0]['predict'][index] = results.tolist()
        # Transform int8 to int
        for index, results in enumerate(dic[key][0]['true']):
            dic[key][0]['true'][index] = [int(val) for val in results]

    # Add list of symptom
    dic['symptoms'] = lt_sympt

    # Save new dictionary
    with open(fout, 'w') as my_file:
        json.dump(dic, my_file)




if __name__=='__main__':
    run()
