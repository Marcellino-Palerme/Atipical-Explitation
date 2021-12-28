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
#import pickle
import numpy as np
import tensorflow as tf

def get_dataset(dir_in, img_size, batch_size, shuffle, seed=None):
    """


    Parameters
    ----------
    dir_in : TYPE
        DESCRIPTION.
    img_size : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.
    shuffle : TYPE
        DESCRIPTION.
    seed : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    create_dataset = tf.keras.preprocessing.image_dataset_from_directory
    return create_dataset(dir_in,
                          shuffle=shuffle,
                          seed=seed,
                          batch_size=batch_size,
                          image_size=img_size)

def extract_label(dataset):
    """
    extract label of tensorflow dataset

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # Take label
    temp = dataset.map(lambda img, lab: lab)
    temp = temp.unbatch()
    # Create list of label
    temp = list(temp.as_numpy_iterator())
    return list(np.array(dataset.class_names)[temp])

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
        - prefix_eval : evaluation from model
        - prefix_loss: loss from model
    """
    dic_pred_true = {}
    # Get labels of image
    dic_pred_true[prefix + '_true'] = extract_label(dataset)
    # Predict label of image
    pred = model.predict(dataset)

    # Take the max of each predition
    pred_max = np.amax(pred, axis=1)
    # Get label of prediction
    pos = [np.where(vals==my_max)[0][0] for vals, my_max in zip(pred,
                                                                pred_max)]
    dic_pred_true[prefix + '_pred'] = [dataset.class_names[index] for index in pos]
    evalut = model.evaluate(dataset)
    dic_pred_true[prefix + '_loss'] = evalut[0]
    dic_pred_true[prefix + '_eval'] = evalut[1]

    return dic_pred_true


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


    train_dataset = get_dataset(train_dir, cst_img_size, cst_batch_size,
                                True, 74)

    validation_dataset = get_dataset(validation_dir, cst_img_size,
                                     cst_batch_size, True, 98)

    test_dataset = get_dataset(test_dir, cst_img_size, cst_batch_size,
                               True, 45)


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

    train_dataset = get_dataset(train_dir, cst_img_size, 1, False)

    validation_dataset = get_dataset(validation_dir, cst_img_size, 1, False)

    test_dataset = get_dataset(test_dir, cst_img_size, 1, False)

    # Create dictionary with pred and true for train/validation/test
    my_dict = pred_true(model, test_dataset, "test")
    my_dict.update(pred_true(model, train_dataset, "train"))
    my_dict.update(pred_true(model, validation_dataset, "val"))

    # Save dictionary
    cst_dict_path = os.path.join(cst_dir_out,
                             cst_my_date + "_pred_true_" + cst_struc + ".json")
    with open(cst_dict_path, 'w') as file:
        json.dump(my_dict, file)

    """
    For tensorflow > 2.3
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
    """

if __name__=='__main__':
    run()
