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
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import pickle

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
accs = []
for index in range(10):
    PATH = "/home/genouest/inra_umr1349/mpalerme/Plant_leave_with_augmentation_" + str(index)
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    test_dir = os.path.join(PATH, 'test')
    BATCH_SIZE = 32
    IMG_SIZE = (132, 132)

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
    print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.EfficientNetB4(input_shape=IMG_SHAPE,
                                                      include_top=False,
                                                      weights='imagenet')

    base_model.trainable = False

    nb_classes =39 
    prediction_layer = tf.keras.layers.Dense(nb_classes, activation="softmax")


    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)

    #x = tf.keras.layers.Flatten()(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history = model.fit(train_dataset,
                        epochs=20,
                        validation_data=validation_dataset,
                        verbose=2)

    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)
    accs.append(accuracy)

print("mean:" + str(np.mean(accs)))
print("std:" + str(np.std(accs)))
print("max:" + str(np.max(accs)))
print("min:" + str(np.min(accs)))
print("var:" + str(np.var(accs)))
print(accs)


"""
predict_label = []
true_label = []

for img, lab in test_dataset:
   # Predict image test
   predict = model.predict(img)

   # Take the max of each predition
   predict_label.append( tf.argmax(predict, axis=1))
   true_label.append(lab)
print(accuracy_score(true_label, predict_label))

# Calculate confusion matrix
conf_mat = confusion_matrix(true_label,
                            predict_label,
                            labels=range(nb_classes))
print(conf_mat)


test_dataset = image_dataset_from_directory(test_dir,
                                            shuffle=True,
                                            batch_size=1,
                                            image_size=IMG_SIZE)

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)


predict_label = []
true_label = []

for img, lab in test_dataset:
   # Predict image test
   predict = model.predict(img)

   # Take the max of each predition
   predict_label.append( tf.argmax(predict, axis=1))
   true_label.append(lab)

# Calculate confusion matrix
conf_mat = confusion_matrix(true_label,
                            predict_label,
                            labels=range(nb_classes))

print(conf_mat)
print(accuracy_score(true_label, predict_label))
"""
