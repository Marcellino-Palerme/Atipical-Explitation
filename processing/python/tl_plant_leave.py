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
from tools_file import create_directory
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="input", help="Directory serial files ")
parser.add_argument("-o", dest="output", help="Directory to save results")
parser.add_argument("-5", dest="ENB5", help="choose EfficientNetB5",
                    action="store_true")
args = parser.parse_args()

# create output directory
create_directory(args.output)

# Define size of image
img_height = 132
img_width = 132

a_img_train = []
a_lab_train = []
a_img_valid = []
a_lab_valid = []
a_img_test = []
a_lab_test = []

for index in range(1,4):
    # open file of serialization
    input_serial = open(join(args.input, "img_train_" + str(index) + "_3.pkl"),
                        "rb")
    # load data
    if a_img_train == []:
        a_img_train = pickle.load(input_serial)
    else:
        a_img_train = np.concatenate((a_img_train, pickle.load(input_serial)),
                                     axis=0)
    # Close file
    input_serial.close() 

for name_var in ["a_lab_train", "a_img_valid", "a_lab_valid", "a_img_test",
                 "a_lab_test"]:
    # open file of serialization
    input_serial = open(join(args.input, name_var + ".pkl"), "rb")
    # load data
    exec(name_var + "= pickle.load(input_serial)")
    # Close file
    input_serial.close()

# Calculate number of classes
nb_classes = len(np.unique(a_lab_train))

if args.ENB5 :
    # Init EfficientNet B5
    EN_BX_conf = EfficientNetB5(weights='imagenet', include_top=False,
                                input_shape=(img_height, img_width, 3))
else :
    # Init EfficientNet B4
    EN_BX_conf = EfficientNetB4(weights='imagenet', include_top=False,
                                input_shape=(img_height, img_width, 3))


# Define the network
model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  EN_BX_conf,
  layers.Dense(nb_classes, activation='softmax')
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
                            labels=range(),
                            normalize='true')
# Save confusion matrix
df_conf_mat = pd.DataFrame(conf_mat)
df_conf_mat.to_csv(join(args.output, 'confusion_matrix_tl_plant_leave.csv'))

class_report = classification_report(a_lab_test, label_pred, output_dict=True)
df_report = pd.DataFrame(class_report).transpose()
df_report.to_csv(join(args.output, 'classification_report_tl_plant_leave.csv'))
