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

    cst_my_date = sys.argv[1]
    cst_struc = sys.argv[2].upper()
    cst_dir_out = os.path.join(os.path.dirname(__file__),
                           "report",
                           cst_my_date + "_tl_atipical_XX")

    cst_path = "/home/genouest/inra_umr1349/mpalerme/dataset_atipical"
    train_dir = os.path.join(cst_path, 'train')
    validation_dir = os.path.join(cst_path, 'validation')
    test_dir = os.path.join(cst_path, 'test')
    cst_batch_size = 32
    cst_img_size = (224, 224)

    # Create file contain all image's name used
    my_file = open(os.path.join(cst_dir_out,
                                cst_my_date + "_dataset_" + cst_struc + ".csv"),
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
                                   batch_size=cst_batch_size,
                                   image_size=cst_img_size)

    validation_dataset = create_dataset(validation_dir,
                                        shuffle=True,
                                        seed=98,
                                        batch_size=cst_batch_size,
                                        image_size=cst_img_size)

    test_dataset = create_dataset(test_dir,
                                  shuffle=True,
                                  seed=45,
                                  batch_size=cst_batch_size,
                                  image_size=cst_img_size)

    # Select structure used
    if re.match(r'^B.$', cst_struc):
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        application = getattr(tf.keras.applications, "EfficientNet" + cst_struc)

    if cst_struc == "INCEPTV3":
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        application = tf.keras.applications.InceptionV3

    if cst_struc == "VGG16":
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        application = tf.keras.applications.VGG16


    # Create the base model from the pre-trained model based on imagenet
    cst_img_shape = cst_img_size + (3,)
    base_model = application(input_shape=cst_img_shape,
                             include_top=False,
                             weights='imagenet')

    # Freeze pre-trained model
    base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable = False


    # Create model
    inputs = tf.keras.Input(shape=cst_img_shape)
    model_proc = preprocess_input(inputs)
    model_proc = base_model(model_proc, training=False)

    # Rebuild top
    model_proc = tf.keras.layers.GlobalAveragePooling2D()(model_proc)
    model_proc = tf.keras.layers.Dropout(0.2)(model_proc)

    nb_classes = len(train_dataset.class_names)
    prediction_layer = tf.keras.layers.Dense(nb_classes, activation='softmax')
    outputs = prediction_layer(model_proc)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(train_dataset,
                        epochs=30,
                        validation_data=validation_dataset,
                        verbose=2)


    # Save history
    cst_hist_path = os.path.join(cst_dir_out, cst_my_date + "_history_" + cst_struc + ".json")
    with open(cst_hist_path, 'w') as file:
        json.dump(history.history, file)

    # Create dictionary with pred and true for train/validation/test
    my_dict = pred_true(model, test_dataset, "test")
    my_dict.update(pred_true(model, train_dataset, "train"))
    my_dict.update(pred_true(model, validation_dataset, "val"))

    # Save dictionary
    cst_dict_path = os.path.join(cst_dir_out,
                             cst_my_date + "_pred_true_" + cst_struc + ".json")
    with open(cst_dict_path, 'w') as file:
        json.dump(my_dict, file)

    # Save model
    cst_model_path = os.path.join(cst_dir_out, cst_my_date + "_" + cst_struc)
    with open(cst_model_path, 'wb') as file:
        pickle.dump({'model':model}, file)

    # Save dataset
    data_name = os.path.join(cst_dir_out, cst_my_date + "_test_" + cst_struc)
    with open(data_name, 'wb') as file:
        pickle.dump({'test':test_dataset}, file)
    data_name = os.path.join(cst_dir_out, cst_my_date + "_train_" + cst_struc)
    with open(data_name, 'wb') as file:
        pickle.dump({'train':train_dataset}, file)
    data_name = os.path.join(cst_dir_out, cst_my_date + "_val_" + cst_struc)
    with open(data_name, 'wb') as file:
        pickle.dump({'validation':validation_dataset}, file)


if __name__=='__main__':
    run()
