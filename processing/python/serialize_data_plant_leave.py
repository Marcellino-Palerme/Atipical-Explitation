#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 17:12:53 2021

@author: mpalerme
"""
import tensorflow as tf
import numpy as np
import argparse
import pickle
from os.path import join
from tools_file import create_directory



parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="input", help="Directory contains images ")
parser.add_argument("-o", dest="output", help="Directory to save results")
args = parser.parse_args()

# create out directory
create_directory(args.output)

# Define size of image
img_height = 132
img_width = 132

# Define directory where take image
data_dir = args.input

# Take all images
ds_full = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed = 159,
    image_size=(img_height, img_width),
    batch_size = 1)

# define number elements of each part
ds_size = len(ds_full)
train_indice = int(0.9 * ds_size)
valid_indice = int(0.07 * ds_size) + train_indice

a_img_train = []
a_lab_train = []
a_img_valid = []
a_lab_valid = []
a_img_test = []
a_lab_test = []

# Split dataset
for index, data in enumerate(ds_full):
    if index < train_indice:
        a_img_train.append(data[0])
        a_lab_train.append(data[1])
    if train_indice <= index < valid_indice:
        a_img_valid.append(data[0])
        a_lab_valid.append(data[1])
    if index >= valid_indice:
        a_img_test.append(data[0])
        a_lab_test.append(data[1])

a_img_train = np.array(a_img_train).reshape((-1,img_height, img_width, 3))
a_lab_train = np.array(a_lab_train).reshape((-1))
a_img_valid = np.array(a_img_valid).reshape((-1, img_height, img_width, 3))
a_lab_valid = np.array(a_lab_valid).reshape((-1))
a_img_test = np.array(a_img_test).reshape((-1, img_height, img_width, 3))
a_lab_test = np.array(a_lab_test).reshape((-1))

part_img_train = int(len(a_img_train)/3)
# create output file of serialization
output = open(join(args.output, "img_train_1_3.pkl"), "wb")
# Save data
pickle.dump(a_img_train[:part_img_train], output)
# Close file
output.close()

# create output file of serialization
output = open(join(args.output, "img_train_2_3.pkl"), "wb")
# Save data
pickle.dump(a_img_train[part_img_train : part_img_train *2 ], output)
# Close file
output.close()

# create output file of serialization
output = open(join(args.output, "img_train_3_3.pkl"), "wb")
# Save data
pickle.dump(a_img_train[part_img_train *2 :], output)
# Close file
output.close()

for name_var in ["a_lab_train", "a_img_valid", "a_lab_valid", "a_img_test",
                 "a_lab_test"]:
    # create output file of serialization
    output = open(join(args.output, name_var + ".pkl"), "wb")
    # Save data
    pickle.dump(eval(name_var), output)
    # Close file
    output.close()