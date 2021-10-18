#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 15:58:25 2021

@author: mpalerme
"""

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cm_print as cmp
import glob
import csv

# Take date when have started script
if len(sys.argv) != 2:
    raise ValueError('Please provide date.')

MY_DATE = sys.argv[1]
DIR_OUT = os.path.join(os.path.dirname(__file__),
                       "report",
                       MY_DATE + "_tl_atipical_B4")

# Save accurancy
results_acc = []
# Save true and predic labels
global_y_true = []
global_y_pred = []

for index in range(10):
    PATH = "/home/genouest/inra_umr1349/mpalerme/dataset_atipical" + str(index)
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    test_dir = os.path.join(PATH, 'test')
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)

    # Create file contain all image's name used
    my_file = open(os.path.join(DIR_OUT, MY_DATE + "_dataset_" + str(index)),
                   "w")
    writer = csv.writer(my_file)
    # Write header
    writer.writerow(["part", "filename"])
    # Write name's part
    writer.writerow(["train"])
    # Write files of train part
    writer.writerows([["", os.path.basename(path)]\
                      for path in glob.glob(os.path.join(train_dir,
                                                         "*",
                                                         "*"))])

    # Write name's part
    writer.writerow(["validation"])
    # Write files of validation part
    writer.writerows([["", os.path.basename(path)]\
                      for path in glob.glob(os.path.join(validation_dir,
                                                         "*",
                                                         "*"))])

    # Write name's part
    writer.writerow(["test"])
    # Write files of test part
    writer.writerows([["", os.path.basename(path)]\
                      for path in glob.glob(os.path.join(test_dir,
                                                         "*",
                                                         "*"))])

    # Close file
    my_file.close()


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

    nb_classes = len(train_dataset.class_names)
    prediction_layer = tf.keras.layers.Dense(nb_classes)


    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)

    #x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(124)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_dataset,
                        epochs=30,
                        validation_data=validation_dataset,
                        verbose=2)

    loss, accuracy = model.evaluate(test_dataset)
    print(accuracy)
    results_acc.append(accuracy)

    y_true = []
    y_pred = []
    for img, label in test_dataset.unbatch():
        # Get label of image
        y_true.append(test_dataset.class_names[label])
        global_y_true.append(test_dataset.class_names[label])

        # Predict label of image
        pred = model.predict(img)

        # Take the max of each predition
        pred_max = np.amax(pred)
        # Get label of prediction
        pos = np.where(pred==pred_max)
        y_pred.append(test_dataset.class_names[pos[0][0]])
        global_y_pred.append(test_dataset.class_names[pos[0][0]])

    cmp.cm_print(os.path.join(DIR_OUT,
                              MY_DATE + "_nor_confusion_matrix_" + str(index)),
                 y_true,
                 y_pred)

    cmp.cm_print(os.path.join(DIR_OUT,
                              MY_DATE + "_confusion_matrix_" + str(index)),
                 y_true,
                 y_pred)


results_acc = np.array(results_acc)
print("moyenne : " + str(results_acc.mean()))
print("min : " + str(results_acc.min()))
print("max : " + str(results_acc.max()))
print("variance : " + str(results_acc.var()))
print("standard deviation  : " + str(results_acc.std()))
# Print global confusion matrix
cmp.cm_print(os.path.join(DIR_OUT,
                          MY_DATE + "_nor_confusion_matrix_global"),
             global_y_true,
             global_y_pred)

cmp.cm_print(os.path.join(DIR_OUT,
                          MY_DATE + "_confusion_matrix_global"),
             global_y_true,
             global_y_pred)
