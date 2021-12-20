#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pickle
import re
import argparse
import time
import itertools as its
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tools_file import file_list, create_directory
from PIL import Image


##############################################################################
### Constants
##############################################################################
train = "train"
validation = "validation"
test = "test"
label = 'label'
recto = 'recto'
verso = 'verso'
symptoms = ['Alt', 'Big', 'Mac', 'Mil', 'Myc', 'Pse', 'Syl']
###############################################################################
### Manage arguments input
###############################################################################
def arguments ():
    """
    manage input arguments

    Returns
    -------
    namespace

    """
    parser = argparse.ArgumentParser()

    # Add argument for source directory
    parser.add_argument('-i', '--dir_in', type=str, help="source directory",
                        required=True, dest='dir_in')

    # Add argument for output directory
    parser.add_argument('-s', '--struct',
                        help="list of pre-training models used",
                        default=['B3', 'B4', 'B5', 'B6', 'VGG16'],
                        dest='lt_struct')


    # Take all arguments
    return parser.parse_args()

##############################################################################
### Additional function
##############################################################################
def select_struct(name_struct):
    """


    Parameters
    ----------
    name_struct : str
        Name of pre-training model.

    Returns
    -------
    None.

    """
    # Select structure used
    if re.match(r'^B.$', name_struct):
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        application = getattr(tf.keras.applications,
                              "EfficientNet" + name_struct)

    if name_struct == "INCEPTV3":
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        application = tf.keras.applications.InceptionV3

    if name_struct == "VGG16":
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        application = tf.keras.applications.VGG16

    return {'pre': preprocess_input, 'app':application}


def pred_true(model, dataset):
    """


    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    dataset : TYPE
        DESCRIPTION.

    Returns
    -------
    dictionnary with two keys by prefix :
        - prefix_pred : list of predicted values by model
        - prefix_true : list of true values

    """
    dic_pred_true = {}
    for prefix in dataset:
        # Get labels of image
        dic_pred_true[prefix + '_true'] = [symptoms[index]\
                                           for index in dataset[prefix][label]]

        # Predict label of image
        pred = model.predict([dataset[prefix][recto], dataset[prefix][verso]])

        # Take the max of each predition
        pred_max = np.amax(pred, axis=1)
        # Get label of prediction
        pos = [np.where(vals==my_max)[0][0] for vals, my_max in zip(pred,
                                                                    pred_max)]
        dic_pred_true[prefix + '_pred'] = [symptoms[index] for index in pos]

    return dic_pred_true

##############################################################################
### Main function
##############################################################################
def run():
    """
    main function

    Returns
    -------
    None.

    """
    # Take input arguments
    args = arguments()

    MY_DATE = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    LT_STRUC = [struc.upper() for struc in args.lt_struct]
    DIR_OUT = os.path.join(os.path.dirname(__file__),
                           "report", MY_DATE + "_multi_view")

    create_directory(DIR_OUT)

    # Define size of image
    img_height = 224
    img_width = 224

    # Define directory where take image
    path = os.path.abspath(os.path.expanduser(args.dir_in))


    dataset = {part:{recto:[], verso:[], label:[]}\
               for part in [train, validation, test]}

    # Take all images and labels
    for part in dataset:
        for index, symptom in enumerate(symptoms):
            files_verso = file_list(os.path.join(path, part, symptom, verso))
            files_recto = file_list(os.path.join(path, part, symptom, recto))
            for file_r, file_v in zip(files_recto, files_verso):
                # Verify we have same leaf
                if file_r[0:8] == file_v[0:8]:
                    try:
                        # Read image recto
                        img_rec = Image.open(os.path.join(path, part, symptom,
                                                          recto, file_r))
                        # Read image verso
                        img_ver = Image.open(os.path.join(path, part, symptom,
                                                          verso, file_v))
                    except IOError :
                        continue

                    dataset[part][recto].append(img_to_array(img_rec))
                    dataset[part][verso].append(img_to_array(img_ver))
                    dataset[part][label].append(index)

        dataset[part][recto] = np.array(dataset[part][recto])
        dataset[part][verso] = np.array(dataset[part][verso])
        dataset[part][label] = np.array(dataset[part][label])


    for strucs in its.product(LT_STRUC, repeat=2):
        # define two input of model
        in_models = []
        inputs = []
        for index_struc, struc in enumerate(strucs):
            # Take all element for model
            info_model = select_struct(struc)

            # Init the  model
            pre_model = info_model['app'](weights='imagenet',
                                          include_top=False,
                                          input_shape=(img_height,
                                                       img_width, 3))
            pre_model._name = struc + str(index_struc)


            # Layer of model isn't trainable
            pre_model.trainable = False
            # Change name to can use twice the same model and freeze
            for layer in pre_model.layers[:]:
                layer._name = layer._name  + str(index_struc)
                layer.trainable = False

            # Define the network
            inputs.append(tf.keras.Input(shape=(img_height, img_width, 3)))
            half_model = info_model['pre'](inputs[-1])
            half_model = pre_model(half_model, training=False)
            # Rebuild top
            half_model = tf.keras.layers.GlobalAveragePooling2D()(half_model)
            half_model = tf.keras.layers.Dropout(0.2)(half_model)

            in_models.append(half_model)

        # Concatenate two input
        concat = tf.keras.layers.concatenate(in_models)

        num_classes = len(symptoms)
        model_final = tf.keras.layers.Dense(num_classes,
                                            activation='softmax')(concat)


        model = tf.keras.Model(inputs=inputs,
                               outputs=model_final)

        # Compile the Network
        base_learning_rate = 0.001
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        # Training the network
        history = model.fit(
                            x=[dataset[train][recto], dataset[train][verso]],
                            y=dataset[train][label],
                            validation_data = ([dataset[validation][recto],
                                                dataset[validation][verso]],
                                               dataset[validation][label]),
                            epochs=30,
                            verbose=0,
                            batch_size=1
                            )

        # Create dictionary with pred and true for train/validation/test
        my_dict = pred_true(model, dataset)

        # Save dictionary
        DICT_FILE = os.path.join(DIR_OUT,
                                 MY_DATE + "_pred_true_"\
                                         + "_".join(strucs) + ".json")
        with open(DICT_FILE, 'w') as file:
            json.dump(my_dict, file)

        # Save history
        HIST_FILE = os.path.join(DIR_OUT,
                                 MY_DATE + "_history_" + "_".join(strucs)\
                                         + ".json")
        with open(HIST_FILE, 'w') as file:
            json.dump(history.history, file)

        # Save dataset
        DATA_NAME = os.path.join(DIR_OUT, MY_DATE + "_dataset_"\
                                 + "_".join(strucs))
        with open(DATA_NAME, 'wb') as file:
            pickle.dump(dataset, file)



if __name__=='__main__':
    run()
