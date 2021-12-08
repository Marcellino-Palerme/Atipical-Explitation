#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 15:58:25 2021

@author: mpalerme
"""

import sys
import os
import glob
import csv
import re
import json
import pickle
import numpy as np
import tensorflow as tf


def pred_true(model, dataset, prefix):
    """


    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    dataset : TYPE
        DESCRIPTION.
    prefix : str
        name of dataset

    Returns
    -------
    dictionnary with two keys :
        - prefix_pred : list of predicted values by model
        - prefix_true : list of true values

    """
    y_true = []
    y_pred = []
    for img, label in dataset.unbatch():
        # Get label of image
        y_true.append(dataset.class_names[label])

        # Predict label of image
        pred = model.predict(img)

        # Take the max of each predition
        pred_max = np.amax(pred)
        # Get label of prediction
        pos = np.where(pred==pred_max)
        y_pred.append(dataset.class_names[pos[0][0]])

    return {prefix + "_pred" : y_pred, prefix + "_true" : y_true}


def run():
    """
    main function

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Take date when have started script
    if len(sys.argv) != 3:
        raise ValueError('Please provide date and/or structure name.')

    MY_DATE = sys.argv[1]
    STRUC = sys.argv[2].upper()
    DIR_OUT = os.path.join(os.path.dirname(__file__),
                           "report",
                           MY_DATE + "_tl_atipical_XX")

    PATH = "/home/genouest/inra_umr1349/mpalerme/dataset_atipical"
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    test_dir = os.path.join(PATH, 'test')
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)

    # Create file contain all image's name used
    my_file = open(os.path.join(DIR_OUT,
                                MY_DATE + "_dataset_" + STRUC + ".csv"),
                   "w")
    writer = csv.writer(my_file)
    # Write header
    writer.writerow(["part", "filename"])
    # Write name's part
    writer.writerow(["train"])
    # Write files of train part
    writer.writerows([["", os.path.basename(path)]\
                      for path in sorted(glob.glob(os.path.join(train_dir,
                                                                "*", "*")))])

    # Write name's part
    writer.writerow(["validation"])
    # Write files of validation part
    writer.writerows([["", os.path.basename(path)]\
                      for path in sorted(glob.glob(os.path.join(validation_dir,
                                                         "*",
                                                         "*")))])

    # Write name's part
    writer.writerow(["test"])
    # Write files of test part
    writer.writerows([["", os.path.basename(path)]\
                      for path in sorted(glob.glob(os.path.join(test_dir,
                                                                "*", "*")))])

    # Close file
    my_file.close()

    create_dataset = tf.keras.preprocessing.image_dataset_from_directory

    train_dataset = create_dataset(train_dir,
                                   shuffle=True,
                                   seed=74,
                                   batch_size=BATCH_SIZE,
                                   image_size=IMG_SIZE)

    validation_dataset = create_dataset(validation_dir,
                                        shuffle=True,
                                        seed=98,
                                        batch_size=BATCH_SIZE,
                                        image_size=IMG_SIZE)

    test_dataset = create_dataset(test_dir,
                                  shuffle=True,
                                  seed=45,
                                  batch_size=BATCH_SIZE,
                                  image_size=IMG_SIZE)

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


    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = application(input_shape=IMG_SHAPE,
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


    # Save history
    HIST_FILE = os.path.join(DIR_OUT, MY_DATE + "_history_" + STRUC + ".json")
    with open(HIST_FILE, 'w') as file:
        json.dump(history.history, file)

    # Create dictionary with pred and true for train/validation/test
    my_dict = pred_true(model, test_dataset, "test")
    my_dict.update(pred_true(model, train_dataset, "train"))
    my_dict.update(pred_true(model, validation_dataset, "val"))

    # Save dictionary
    DICT_FILE = os.path.join(DIR_OUT,
                             MY_DATE + "_pred_true_" + STRUC + ".json")
    with open(DICT_FILE, 'w') as file:
        json.dump(my_dict, file)

    # Save model
    MODEL_NAME = os.path.join(DIR_OUT, MY_DATE + "_" + STRUC)
    with open(MODEL_NAME, 'wb') as file:
        pickle.dump({'model':model}, file)

    # Save dataset
    DATA_NAME = os.path.join(DIR_OUT, MY_DATE + "_test_" + STRUC)
    with open(DATA_NAME, 'wb') as file:
        pickle.dump({'test':test_dataset}, file)
    DATA_NAME = os.path.join(DIR_OUT, MY_DATE + "_train_" + STRUC)
    with open(DATA_NAME, 'wb') as file:
        pickle.dump({'train':train_dataset}, file)
    DATA_NAME = os.path.join(DIR_OUT, MY_DATE + "_val_" + STRUC)
    with open(DATA_NAME, 'wb') as file:
        pickle.dump({'validation':validation_dataset}, file)


if __name__=='__main__':
    run()
