#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:33:00 2021

@author: port-mpalerme
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB5, EfficientNetB4
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from os.path import join
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="input", help="Directory contains images ")
parser.add_argument("-o", dest="output", help="Directory to save results")
parser.add_argument("-5", dest="ENB5", help="choose EfficientNetB5",
                    action="store_true")
args = parser.parse_args()

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

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

a_img_train = np.array(a_img_train).reshape((-1, img_height, img_width, 3))
a_lab_train = np.array(a_lab_train).reshape((-1))
a_img_valid = np.array(a_img_valid).reshape((-1, img_height, img_width, 3))
a_lab_valid = np.array(a_lab_valid).reshape((-1))
a_img_test = np.array(a_img_test).reshape((-1, img_height, img_width, 3))
a_lab_test = np.array(a_lab_test).reshape((-1))

if args.ENB5 :
    # Init EfficientNet B5
    EN_BX_conf = EfficientNetB5(weights='imagenet', include_top=False,
                                input_shape=(img_height, img_width, 3))
else :
    # Init EfficientNet B4
    EN_BX_conf = EfficientNetB4(weights='imagenet', include_top=False,
                                input_shape=(img_height, img_width, 3))

# training all the layers
for layer in EN_BX_conf.layers[:]:
    layer.trainable = True

num_classes = len(ds_full.class_names)

# Define the network
model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  EN_BX_conf,
  layers.Flatten(),
  layers.Dense(num_classes, activation='softmax')
])

# Compile the Network
model.compile(
  optimizer='adam',
  loss= 'sparse_categorical_crossentropy',
  metrics=['accuracy'])

#Callback
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=join(args.output, 'tl_plant_leave.hd5'),
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False)

# Training the network
model.fit(
  a_img_train,
  a_lab_train,
  validation_data = (a_img_valid, a_lab_valid),
  epochs=30,
  verbose=2,
  callbacks=[model_checkpoint_callback],
  batch_size = 32
)

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model(join(args.output, 'tl_plant_leave.hd5'))

# Predict image test
predict = model.predict(a_img_test)

# Take the max of each predition
pred_max = np.amax(predict, axis=1)

# Extract the predict class of each max
label_pred = []
for pred , our_max in zip(predict, pred_max):
    label_pred.append(np.where(pred==our_max)[0][0])

# Calculate confusion matrix
conf_mat = confusion_matrix(a_lab_test, label_pred,
                            labels=range(len(ds_full.class_names)),
                            normalize='true')
# Save confusion matrix
df_conf_mat = pd.DataFrame(conf_mat)
df_conf_mat.to_csv(join(args.output, 'confusion_matrix_tl_plant_leave.csv'))

class_report = classification_report(a_lab_test, label_pred, output_dict=True)
df_report = pd.DataFrame(class_report).transpose()
df_report.to_csv(join(args.output, 'classification_report_tl_plant_leave.csv'))
