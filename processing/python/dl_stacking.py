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
import sklearn.ensemble as ens
from tools_file import create_directory



##############################################################################
### Constants
##############################################################################
CST_DL = "dl"
CST_STACK = "stacking"
CST_TRAIN = "train"
CST_VAL = "validation"
CST_TEST = "test"
CST_LAB = 'label'
CST_RECTO = 'recto'
CST_VERSO = 'verso'
CST_SYMP = ['Alt', 'Big', 'Mac', 'Mil', 'Myc', 'Pse', 'Syl']
CST_HIST = 'history'
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
def write_files(writer, dir_r, dir_v):
    """
    Write all file of recto and verso directory

    Parameters
    ----------
    writer : TYPE
        csv writer.
    dir_r : str
        path of recto directory contain list symptom directory.
    dir_v : str
        path of verso directory contain list symptom directory.

    Returns
    -------
    None.

    """
    lt_files = []
    for face in [dir_r, dir_v]:
        # Take all files
        lt_files += [os.path.basename(path)\
                     for path in glob.glob(os.path.join(face, "*", "*"))]
    # Write files of part
    writer.writerows([["", path] for path in sorted(lt_files)])


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

        # Deep learning part
        for part in [CST_TRAIN, CST_VAL]:
            # Write name's part
            writer.writerow([CST_DL + '_' + part])
            # Write files
            write_files(writer,
                        os.path.join(dir_r, CST_DL, part),
                        os.path.join(dir_v, CST_DL, part))

        # Stacking part
        # Take all split
        lt_split = sorted(os.listdir(os.path.join(dir_r, CST_STACK)))
        for split in lt_split:
            for part in [CST_TRAIN, CST_TEST]:
                # Write name's part
                writer.writerow([CST_STACK + '_' + part])
                # Write files
                write_files(writer,
                            os.path.join(dir_r, CST_STACK, split, part),
                            os.path.join(dir_v, CST_STACK, split, part))


def get_dataset(path_recto, path_verso, image_size, batch_size, shuffle,
                seed=None):
    """
    Get dataset from recto and verso path

    Parameters
    ----------
    path_recto : TYPE
        DESCRIPTION.
    path_verso : TYPE
        DESCRIPTION.
    image_size : TYPE
        DESCRIPTION.
    batch_size : TYPE
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
    dict
        DESCRIPTION.

    """

    create_dataset = tf.keras.preprocessing.image_dataset_from_directory

    temp_recto = create_dataset(path_recto,
                                shuffle=shuffle,
                                batch_size=batch_size,
                                image_size=image_size,
                                seed=seed)
    temp_verso = create_dataset(path_verso,
                                shuffle=shuffle,
                                batch_size=batch_size,
                                image_size=image_size,
                                seed=seed)

    # Verify label names and order
    if ((temp_recto.class_names != CST_SYMP) or
        (extract_label(temp_recto) != extract_label(temp_verso))):
        raise Exception('Differnce between labels')

    return {CST_RECTO:temp_recto, CST_VERSO:temp_verso}


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

def def_model(struct, img_shape):
    """
    define model

    Parameters
    ----------
    struct : list of str
        list of pre-training model name.
    img_shape : tupple
        image shape.

    Returns
    -------
    model : keras model
        DESCRIPTION.

    """
    # Take all element for model
    info_model = select_struct(struct)

    # Init the  model
    pre_model = info_model['app'](weights='imagenet',
                                  include_top=False,
                                  input_shape=img_shape)

    # Layer of model isn't trainable
    pre_model.trainable = False
    # freeze
    for layer in pre_model.layers[:]:
        layer.trainable = False

    # Define the network
    inputs = tf.keras.Input(shape=img_shape)
    model = info_model['pre'](inputs)
    model = pre_model(model, training=False)

    model = tf.keras.layers.GlobalAveragePooling2D()(model)

    # new top
    model = tf.keras.layers.Dropout(0.2)(model)

    num_classes = len(CST_SYMP)
    model = tf.keras.layers.Dense(num_classes, activation='softmax')(model)

    model = tf.keras.Model(inputs=inputs, outputs=model)

    # Compile the Network
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


def fit_model(model, dataset_train, dataset_val):
    """
    fit the model

    Parameters
    ----------
    model : keras model
        DESCRIPTION.
    dataset_train : tensor_dataset
        image to train.
    dataset_val : tensor_dataset
        image to validate.

    Returns
    -------
    list
        fited model and fiting's history.

    """
    history = model.fit(x=dataset_train,
                        validation_data=dataset_val,
                        epochs=3,
                        verbose=0)

    return [model, history]

def def_n_fit_model(lt_struct, dataset, img_shape):
    """
    define model and fiting

    Parameters
    ----------
    lt_struct : list of str
        list of pre-training model name.
    dataset : dictionary of tensor_dataset
        DESCRIPTION.
    img_shape : tupple
        image shape.

    Returns
    -------
    models : dictionary of fiting keras model
        DESCRIPTION.

    """

    models = {}

    for struct in lt_struct:
        models[struct] = {}
        for face in [CST_RECTO, CST_VERSO]:
            # define model
            models[struct][face] = def_model(struct, img_shape)
            # fit model
            temp = fit_model(models[struct][face],
                             dataset[CST_TRAIN][face],
                             dataset[CST_VAL][face])
            models[struct][face] = temp[0]
            models[struct][face + '-' + CST_HIST] = temp[1]

    return models


def get_split_dataset(path_recto, path_verso, img_size):
    """
    Get data from k-fold split

    Parameters
    ----------
    path_recto : str
        path of recto directory.
    path_verso : str
        path of verso directory.
    img_size : tupple of int
        image size (width, height).

    Returns
    -------
    dataset : list of dictionary
        One case by split. In each case, there is a dictionary.
        dictionary = {'train':{'recto':tensor, 'verso':tensor},
                      'test':{'recto':tensor, 'verso':tensor}}

    """

    dataset = []
    # Take all split
    lt_split = sorted(os.listdir(os.path.join(path_recto, CST_STACK)))
    for split in lt_split:
        dataset.append({})
        for part in [CST_TRAIN, CST_TEST]:
            dataset[-1][part].update((get_dataset(os.path.join(path_recto,
                                                               CST_STACK,
                                                               split, part),
                                                  os.path.join(path_verso,
                                                               CST_STACK,
                                                               split, part),
                                                  img_size, 32,False)))

    return dataset

def get_predict_dataset(lt_struct, models, in_dataset):
    """
    Get predict value from training models

    Parameters
    ----------
    lt_struct : list of str
        Name of training models.
    models : dictionary of training keras models
        DESCRIPTION.
    in_dataset : list of dictionary
        all images (tensor) of k-fold split.

    Returns
    -------
    temp : list of dictionary
        One case by split. In each case, there is a dictionary.
        {'label':{'train':[labels], 'test':[labels]},
         'Model_Name': {'train':{'recto':[predicts], 'verso':[predicts]},
                        'test':{'recto':[predicts], 'verso':[predicts]}},
         ...}

    """

    temp = []
    for data in in_dataset:
        temp.append({})
        # extract true label
        temp[-1][CST_LAB] = {CST_TRAIN:extract_label(data[CST_TRAIN]
                                                         [CST_VERSO]),
                             CST_TEST:extract_label(data[CST_TEST]
                                                        [CST_VERSO])}
        for struct in lt_struct:
            temp[-1][struct]={}
            for part in [CST_TRAIN, CST_TEST]:
                temp[-1][struct][part]={}
                for index, face in enumerate([CST_RECTO, CST_VERSO]):
                    # Get predict
                    temp[-1][struct][part][face] = models[struct[index]][face].\
                                                          predict(data[part][face])

    return temp


def stacking_fit_pred(name_recto, name_verso, dataset, out_file):

    result = []
    for data in dataset:
        result.append({})
        feature = {}
        for part in [CST_TRAIN, CST_TEST]:
            # Concatenate recto and verso features
            feature[part] = np.concatenate((data[name_recto][part][CST_RECTO],
                                            data[name_verso][part][CST_VERSO]))
            # Save true values
            result[-1][part + '_true'] = data[CST_LAB][part]

        # Fit classifier
        clf = ens.GradientBoostingClassifier().fit(feature[CST_TRAIN],
                                                   data[CST_LAB][CST_TRAIN])

        # Predict
        for part in [CST_TRAIN, CST_TEST]:
            result[-1][part + '_pred'] = clf.predict(feature[part])
            result[-1][part + '_eval'] = clf.score(feature[part],
                                                   result[-1][part + '_true'])

    # Calculate global acc
    mean = np.mean([acc[CST_TEST + '_eval'] for acc in result])
    print(name_recto + '-' + name_verso + ' : ' + str(mean))

    # TODO: save results

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
        dic_pred_true[prefix + '_pred'] = [CST_SYMP[index] for index in pos]

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
    img_height = 224
    img_width = 224

    # Define directory where take image
    path_recto = os.path.abspath(os.path.expanduser(args.dir_rec))
    path_verso = os.path.abspath(os.path.expanduser(args.dir_ver))

    # Save Dataset
    save_dataset(path_recto, path_verso, cst_dir_out)

    # Get dataset for deep learning
    dataset = {}
    for part in [CST_TRAIN, CST_VAL]:
        dataset[part] = get_dataset(os.path.join(path_recto, CST_DL, part),
                                    os.path.join(path_verso, CST_DL, part),
                                    (img_width, img_height), 32, True, 87)

    # Fit all model
    models = def_n_fit_model(cst_lt_struct, dataset, (img_width, img_height, 3))


    # TODO : Finish staking part
    # Get dataset from k-fold split
    dataset = get_split_dataset(path_recto, path_verso, (img_width,
                                                         img_height))

    # Get predict dataset from model
    dataset = get_predict_dataset(cst_lt_struct, models, dataset)

            # TODO: Get label train + get Test predict and label
            #       learn stacking + write predict
    # Save feature data predict
    out_path = os.path.join(cst_dir_out, cst_date + "_feature_pred.json")
    with open(out_path, 'w') as out_file:
        json.dump(dataset, out_file)

    for strucs in its.product(cst_lt_struct, repeat=2):

        print(model.evaluate(dataset[CST_TEST]))
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
