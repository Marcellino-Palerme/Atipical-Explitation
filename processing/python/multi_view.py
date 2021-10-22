#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tools_file import file_list
from os.path import join
from PIL import Image
from os import listdir, makedirs, rmdir, remove
from os.path import isfile, join, exists, splitext
import json
import sys
import os
import re


# Take date when have started script
if len(sys.argv) != 3:
    raise ValueError('Please provide date and/or structure name.')

MY_DATE = sys.argv[1]
STRUC = sys.argv[2].upper()
DIR_OUT = os.path.join(os.path.dirname(__file__),
                       "report",
                       MY_DATE + "_tl_atipical_" + STRUC)


# Define size of image
img_height = 224
img_width = 224

# Define directory where take image
path = '/home/genouest/inra_umr1349/mpalerme/dataset_atipical'
symptoms = ['Alt', 'Big', 'Mac', 'Mil', 'Myc', 'Pse', 'Syl']
recto = 'recto'
verso = 'verso'

train_photos_recto = []
train_photos_verso = []
train_labels = []
valid_photos_recto = []
valid_photos_verso = []
valid_labels = []
test_photos_recto = []
test_photos_verso = []
test_labels = []

# Take all images and labels
for label, symptom in enumerate(symptoms):
    for split in ['train', 'validation', 'test']:
        files_verso = file_list(join(path, split, symptom, verso))
        files_recto = file_list(join(path, split, symptom, recto))
        for file_r, file_v in zip(files_recto, files_verso):
            try:
                # Read image recto
                img_rec = Image.open(join(path, split, symptom, recto, file_r))
                # Read image verso
                img_ver = Image.open(join(path, split, symptom, verso, file_v))
            except IOError :
                continue
            if split == 'train':
                # Transform image to array and add array
                train_photos_recto.append(img_to_array(img_rec))
                train_photos_verso.append(img_to_array(img_ver))
                # Add Image's label
                train_labels.append(label)
            if split == 'test':
                # Transform image to array and add array
                test_photos_recto.append(img_to_array(img_rec))
                test_photos_verso.append(img_to_array(img_ver))
                # Add Image's label
                test_labels.append(label)
            if split == 'validation':
                # Transform image to array and add array
                valid_photos_recto.append(img_to_array(img_rec))
                valid_photos_verso.append(img_to_array(img_ver))
                # Add Image's label
                valid_labels.append(label)

train_photos_recto = np.array(train_photos_recto)
train_photos_verso = np.array(train_photos_verso)
train_labels = np.array(train_labels)
test_photos_recto = np.array(test_photos_recto)
test_photos_verso = np.array(test_photos_verso)
test_labels = np.array(test_labels)
valid_photos_recto = np.array(valid_photos_recto)
valid_photos_verso = np.array(valid_photos_verso)
valid_labels = np.array(valid_labels)
print(train_photos_recto)


# Select structure used
if re.match(r'^B.$', STRUC):
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    application = getattr(tf.keras.applications, "EfficientNet" + STRUC)

if STRUC == "INCEPTV3":
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input
    application = tf.keras.applications.InceptionV3

if STRUC == "VGG16":
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    application = tf.keras.applications.VGG16


# Define the process for augmentation data
data_augmentation_r = tf.keras.Sequential([
     tf.keras.Input((img_height, img_width, 3)),
     tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
     tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
     tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)])

data_augmentation_v = tf.keras.Sequential([
     tf.keras.Input((img_height, img_width, 3)),
     tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
     tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
     tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)])

# Init the VGG model
vgg_conv_r = application(weights='imagenet', include_top=False,
                         input_shape=(img_height, img_width, 3))
vgg_conv_r._name = "vgg16_r"

vgg_conv_v = application(weights='imagenet', include_top=False,
                         input_shape=(img_height, img_width, 3))
vgg_conv_v._name = "vgg16_v"


# Layer of vgg isn't trainable
vgg_conv_v.trainable = False
vgg_conv_r.trainable = False
# Change name to can use twice the same model
for layer in vgg_conv_r.layers[:]:
    layer._name = layer._name  + str("_2")

# Define the network
# create parallel models
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
model_verso = preprocess_input(inputs)
model_verso = vgg_conv_v(model_verso)
# Rebuild top
model_verso = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model_verso)
model_verso = tf.keras.layers.BatchNormalization()(model_verso)

top_dropout_rate = 0.2
model_verso = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(model_verso)

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
model_recto = preprocess_input(inputs)
model_recto = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool_recto")(model_recto)
model_recto = tf.keras.layers.BatchNormalization()(model_recto)

top_dropout_rate = 0.2
model_recto = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout_recto")(model_recto)

concat = tf.keras.layers.concatenate([model_recto,model_verso])

num_classes = len(symptoms)
model_final = tf.keras.layers.Dense(num_classes, activation='softmax')(concat)


model = tf.keras.Model(inputs=[data_augmentation_r.input,
                               data_augmentation_v.input],
                       outputs=model_final)

"""
#Callback
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='/home/genouest/inra_umr1349/mpalerme/model_multi_view_best.hd5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
"""
# Compile the Network
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
print(model.summary())

global logs

# Training the network
history = model.fit(
                    x= [train_photos_recto, train_photos_verso], y=train_labels,
                    validation_data = ([valid_photos_recto, valid_photos_verso], valid_labels),
                    epochs=30,
                    verbose=2,
                    batch_size=1
                    )
"""
# Save the network
model.save("/home/genouest/inra_umr1349/mpalerme/model_multi_view_last.hd5")
"""
print(model.evaluate([test_photos_recto, test_photos_verso]))

# Save history
HIST_FILE = os.path.join(DIR_OUT, MY_DATE + "_history.json")
with open(HIST_FILE, 'w') as file:
    json.dump(history.history, file)

