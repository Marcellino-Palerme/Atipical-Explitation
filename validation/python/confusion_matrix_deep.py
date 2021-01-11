#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 07:25:56 2020

@author: port-mpalerme
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


batch_size = 32
# Define size of image
img_height = 224
img_width = 224

# Define directory where take image
# In directoty there is one directory by class
data_dir = "/home/port-mpalerme/Documents/Atipical/Traitement/photos/deep"
data_test = data_dir + "/test"

# Take images for validation (test)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_test,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=False)

# Load the network
model = keras.models.load_model('/home/port-mpalerme/Documents/Atipical/Traitement/model.hd5')

# predict the validation (test) images
predictions = model.predict(test_ds)

# Take the max of each predition
pred_max = np.amax(predictions,axis=1)

# Extract the predict class of each max
label_pred = []
for pred , our_max in zip(predictions, pred_max):
    label_pred.append(np.where(pred==our_max)[0][0])

print(label_pred)

# Extract true class of each image
label_true = np.array([])
for data in test_ds:
    label_true = np.concatenate((label_true, data[1]))

print(label_true)

# Show accuracy
print(accuracy_score(label_true, label_pred))
# Show confusion matrix
print(confusion_matrix(label_true, label_pred, normalize='true'))
        

