#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
# import pickle
import re
import csv
import glob
import argparse
import time
import itertools as its
import tensorflow as tf
import numpy as np
from tools_file import create_directory



##############################################################################
### Constants
##############################################################################
CST_TRAIN = "train"
CST_VAL = "validation"
CST_TEST = "test"
CST_LAB = 'label'
CST_RECTO = 'recto'
CST_VERSO = 'verso'
CST_SYMP = ['Alt', 'Big', 'Mac', 'Mil', 'Myc', 'Pse', 'Syl']
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
def save_dataset(dir_r, dir_v, dir_out):
    """
    Create csv with all images names of each part of dataset

    Parameters
    ----------
    dir_r : TYPE
        DESCRIPTION.
    dir_v : TYPE
        DESCRIPTION.
    dir_out : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Create file contain all image's name used
    with open(os.path.join(dir_out,"dataset.csv"),
              "w") as my_file:
        writer = csv.writer(my_file)
        # Write header
        writer.writerow(["part", "filename"])

        for part in [CST_TRAIN, CST_VAL, CST_TEST]:
            # Write name's part
            writer.writerow([part])
            lt_files = []
            for face in [dir_r, dir_v]:
                # Take all files
                lt_files += [os.path.basename(path)\
                             for path in glob.glob(os.path.join(face, part,
                                                                "*", "*"))]
            # Write files of part
            writer.writerows([["", path] for path in sorted(lt_files)])


def get_dataset(dir_r, dir_v, img_size, shuffle, seed=None):
    """


    Parameters
    ----------
    dir_r : TYPE
        DESCRIPTION.
    dir_v : TYPE
        DESCRIPTION.
    img_size : TYPE
        DESCRIPTION.
    shuffle : TYPE
        DESCRIPTION.
    seed : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    dataset : TYPE
        DESCRIPTION.

    """
    create_dataset = tf.keras.preprocessing.image_dataset_from_directory
    temp_recto = create_dataset(dir_r,
                                shuffle=shuffle,
                                batch_size=1,
                                image_size=img_size,
                                seed=seed)

    temp_verso = create_dataset(dir_v,
                                shuffle=shuffle,
                                batch_size=1,
                                image_size=img_size,
                                seed=seed)

    # Verify label names and order
    if ((temp_recto.class_names != CST_SYMP) or
        (extract_label(temp_recto) != extract_label(temp_verso))):
        raise Exception('Differnce between labels')

    dataset = tf.data.Dataset.zip((temp_recto,
                                   temp_verso))

    dataset = dataset.map(lambda rec, ver: ({'layer_0':rec[0],
                                             'layer_1':ver[0]},
                                            rec[1]))
    return dataset

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
    return list(np.array(CST_SYMP)[temp])

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
        - prefix_eval : evaluation from model
        - prefix_loss: loss from model

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
        dic_pred_true[prefix + '_pred'] = [CST_SYMP[index] for index in pos]
        evalut = model.evaluate(dataset[prefix])
        dic_pred_true[prefix + '_loss'] = evalut[0]
        dic_pred_true[prefix + '_eval'] = evalut[1]

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

    cst_date = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    cst_lt_struct = [struc.upper() for struc in args.lt_struct]
    cst_dir_out = os.path.join(os.path.dirname(__file__),
                               "report", cst_date + "_multi_view")

    create_directory(cst_dir_out)

    # Define size of image
    cst_img_size = (224, 224)
    cst_img_shape = cst_img_size + (3,)

    # Define directory where take image
    path_recto = os.path.abspath(os.path.expanduser(args.dir_rec))
    path_verso = os.path.abspath(os.path.expanduser(args.dir_ver))

    # Save Dataset
    save_dataset(path_recto, path_verso, cst_dir_out)

    dataset = {}

    # Take all images and labels
    for part in [CST_TRAIN, CST_VAL]:
        dataset[part] = get_dataset(os.path.join(path_recto, part),
                                    os.path.join(path_verso, part),
                                    cst_img_size, True, 33)


    for strucs in its.product(cst_lt_struct, repeat=2):
        # define two input of model
        in_models = []
        inputs = []
        for index_struc, struc in enumerate(strucs):
            # Take all element for model
            info_model = select_struct(struc)

            # Init the  model
            pre_model = info_model['app'](weights='imagenet',
                                          include_top=False,
                                          input_shape=cst_img_shape)
            pre_model._name = struc + str(index_struc)


            # Layer of model isn't trainable
            pre_model.trainable = False
            # Change name to can use twice the same model and freeze
            for layer in pre_model.layers[:]:
                layer._name = layer._name  + str(index_struc)
                layer.trainable = False

            # Define the network
            inputs.append(tf.keras.Input(shape=cst_img_shape,
                                         name='layer_' + str(index_struc)))
            half_model = info_model['pre'](inputs[-1])
            half_model = pre_model(half_model, training=False)

            half_model = tf.keras.layers.GlobalAveragePooling2D()(half_model)
            in_models.append(half_model)

        # Concatenate two input
        concat = tf.keras.layers.concatenate(in_models)

        # new top
        concat = tf.keras.layers.Dropout(0.2)(concat)

        num_classes = len(CST_SYMP)
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
                            x=dataset[CST_TRAIN],
                            validation_data=dataset[CST_VAL],
                            epochs=30,
                            verbose=0,
                            batch_size=1
                            )


        # Take all images and labels
        for part in [CST_TRAIN, CST_VAL, CST_TEST]:
            dataset[part] = get_dataset(os.path.join(path_recto, part),
                                        os.path.join(path_verso, part),
                                        cst_img_size, False)

        # Create dictionary with pred and true for train/validation/test
        my_dict = pred_true(model, dataset)

        # Save dictionary
        dict_file = os.path.join(cst_dir_out,
                                 cst_date + "_pred_true_"\
                                         + "_".join(strucs) + ".json")
        with open(dict_file, 'w') as file:
            json.dump(my_dict, file)

        # Save history
        hist_file = os.path.join(cst_dir_out,
                                 cst_date + "_history_" + "_".join(strucs)\
                                         + ".json")
        with open(hist_file, 'w') as file:
            json.dump(history.history, file)

        """
        # When we can use tensorflow with pickle
        # Save dataset
        DATA_NAME = os.path.join(cst_dir_out, cst_date + "_dataset_"\
                                 + "_".join(strucs))
        with open(DATA_NAME, 'wb') as file:
            pickle.dump(dataset, file)
        """


if __name__=='__main__':
    run()
