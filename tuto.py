#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 15:58:25 2021

@author: mpalerme
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


from tensorflow.keras.preprocessing import image_dataset_from_directory
"""
PATH = "/home/genouest/inra_umr1349/mpalerme/a_test/test"
train_dir = PATH 
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.1,
                                             subset="training",
                                             seed=159
                                             )

validation_dataset = image_dataset_from_directory(train_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE,
                                                  validation_split=0.1,
                                                  subset="validation",
                                                  seed=159
                                                  )
"""
"""
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
"""

PATH = "/home/genouest/inra_umr1349/mpalerme/mini_plant_div"
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

test_dataset = image_dataset_from_directory(test_dir,
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)
"""
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(3*(val_batches // 10))
validation_dataset = validation_dataset.skip(3*(val_batches // 10))
"""
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
nb_classes =3 
prediction_layer = tf.keras.layers.Dense(nb_classes, activation="softmax")


inputs = tf.keras.Input(shape=(160, 160, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=validation_dataset,
                    verbose=2)

# Predict image test
predict = model.predict(list(test_dataset.as_numpy_iterator())[0][0])

# Take the max of each predition
pred_max = np.amax(predict, axis=1)

# Extract the predict class of each max
label_pred = []
for pred , our_max in zip(predict, pred_max):
    label_pred.append(np.where(pred==our_max)[0][0])

# Calculate confusion matrix
conf_mat = confusion_matrix(list(test_dataset.as_numpy_iterator())[0][1],
                            label_pred,
                            labels=range(nb_classes),
                            normalize='true')
print(conf_mat)
