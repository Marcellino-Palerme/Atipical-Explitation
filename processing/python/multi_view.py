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
import numpy as np
from tools_file import create_directory



##############################################################################
### Constants
##############################################################################
cst_train = "train"
cst_val = "validation"
cst_test = "test"
cst_lab = 'label'
cst_recto = 'recto'
cst_verso = 'verso'
cst_symp = ['Alt', 'Big', 'Mac', 'Mil', 'Myc', 'Pse', 'Syl']
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
    parser.add_argument('-r', '--dir_rec', type=str,
                        help="source recto directory",
                        required=True, dest='dir_rec')

    # Add argument for source directory
    parser.add_argument('-v', '--dir_ver', type=str,
                        help="source verso directory",
                        required=True, dest='dir_ver')

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
    temp = dataset.map(lambda img, lab: lab[0])
    # Create list of label
    temp = list(temp.as_numpy_iterator())
    return list(np.array(cst_symp)[temp])

def pred_true(model, dataset):
    """
    create dictionary with predict and true for tran/test/validation part

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
        dic_pred_true[prefix + '_true'] = extract_label(dataset[prefix])
        # Predict label of image
        pred = model.predict(dataset[prefix])

        # Take the max of each predition
        pred_max = np.amax(pred, axis=1)
        # Get label of prediction
        pos = [np.where(vals==my_max)[0][0] for vals, my_max in zip(pred,
                                                                    pred_max)]
        dic_pred_true[prefix + '_pred'] = [cst_symp[index] for index in pos]

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
    path_recto = os.path.abspath(os.path.expanduser(args.dir_rec))
    path_verso = os.path.abspath(os.path.expanduser(args.dir_ver))


    dataset = {part:{cst_recto:[], cst_verso:[], cst_lab:[]}\
               for part in [cst_train, cst_val, cst_test]}

    create_dataset = tf.keras.preprocessing.image_dataset_from_directory
    # Take all images and labels
    for part in dataset:
        temp_recto = create_dataset(os.path.join(path_recto, part),
                                    shuffle=False,
                                    batch_size=1,
                                    image_size=(224, 224))
        temp_verso = create_dataset(os.path.join(path_verso, part),
                                    shuffle=False,
                                    batch_size=1,
                                    image_size=(224, 224))

        # Verify label names and order
        if ((temp_recto.class_names != cst_symp) or
            (extract_label(temp_recto) != extract_label(temp_verso))):
            raise Exception('Differnce between labels')

        dataset[part] = tf.data.Dataset.zip((temp_recto,
                                             temp_verso))

        dataset[part] = dataset[part].map(lambda rec, ver: ({'layer_0':rec[0],
                                                             'layer_1':ver[0]},
                                                             rec[1]))


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
            inputs.append(tf.keras.Input(shape=(img_height, img_width, 3),
                                         name='layer_' + str(index_struc)))
            half_model = info_model['pre'](inputs[-1])
            half_model = pre_model(half_model, training=False)
            # Rebuild top
            half_model = tf.keras.layers.GlobalAveragePooling2D()(half_model)
            half_model = tf.keras.layers.Dropout(0.2)(half_model)

            in_models.append(half_model)

        # Concatenate two input
        concat = tf.keras.layers.concatenate(in_models)

        num_classes = len(cst_symp)
        model_final = tf.keras.layers.Dense(num_classes,
                                            activation='softmax')(concat)


        model = tf.keras.Model(inputs=inputs,
                               outputs=model_final)

        # Compile the Network
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        # Training the network
        history = model.fit(
                            x=dataset[cst_train],
                            validation_data=dataset[cst_val],
                            epochs=1,
                            verbose=0,
                            batch_size=1
                            )

        print(model.evaluate(dataset[cst_test]))
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

        """
        # When we can use tensorflow with pickle
        # Save dataset
        DATA_NAME = os.path.join(DIR_OUT, MY_DATE + "_dataset_"\
                                 + "_".join(strucs))
        with open(DATA_NAME, 'wb') as file:
            pickle.dump(dataset, file)
        """


if __name__=='__main__':
    run()
