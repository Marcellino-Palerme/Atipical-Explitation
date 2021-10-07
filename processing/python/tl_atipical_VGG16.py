#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 15:58:25 2021

@author: mpalerme
"""

import numpy as np
import os
import tensorflow as tf


from tensorflow.keras.preprocessing import image_dataset_from_directory

results = []

for index in range(10):
    PATH = "/home/genouest/inra_umr1349/mpalerme/dataset_atipical" + str(index)
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    test_dir = os.path.join(PATH, 'test')
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)

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

    preprocess_input = tf.keras.applications.vgg16.preprocess_input

    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    # Create the base model from the pre-trained model imagenet 
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
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
    results.append(accuracy)


results = np.array(results)
print("moyenne : " + str(results.mean()))
print("min : " + str(results.min()))
print("max : " + str(results.max()))
print("variance : " + str(results.var()))
print("standard deviation  : " + str(results.std()))


