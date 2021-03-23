#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from keras import layers
from keras.applications import vgg16, vgg19
import numpy as np
from tools_file import file_list
from os.path import join
from PIL import Image
from sklearn.model_selection import train_test_split


# Define size of image
img_height = 224
img_width = 224

# Define directory where take image
Big = '/home/genouest/inra_umr1349/mpalerme/deep/photos/Big_bdb_cut2_resize'
Mac = '/home/genouest/inra_umr1349/mpalerme/deep/photos/Mac_bdb_cut2_resize'
Myc = '/home/genouest/inra_umr1349/mpalerme/deep/photos/Myc_bdb_cut2_resize'
Pse = '/home/genouest/inra_umr1349/mpalerme/deep/photos/Pse_bdb_cut2_resize'
Syl = '/home/genouest/inra_umr1349/mpalerme/deep/photos/Syl_bdb_cut2_resize'
recto = 'recto'
verso = 'verso'

train_photos_recto = []
train_photos_verso = []
train_labels = []
test_photos_recto = []
test_photos_verso = []
test_labels = []

# Take all images and labels
for label, where in enumerate([Big, Mac, Myc, Pse, Syl]):
    files_verso = file_list(join(where, verso))
    files_recto = file_list(join(where, recto))
    index = 0
    train_test = train_test_split(range(len(files_verso)),
                                  test_size=0.3)
    for file_r, file_v in zip(files_recto, files_verso):
        try:
            # Read image recto
            img_rec = Image.open(join(where, recto, file_r))
            # Read image verso
            img_ver = Image.open(join(where, verso, file_v))
        except IOError :
            index = index + 1
            continue
        if index in train_test[0]:
            # Transform image to array and add array
            train_photos_recto.append(img_to_array(img_rec))
            train_photos_verso.append(img_to_array(img_ver))
            # Add Image's label
            train_labels.append(label)
        else:
            # Transform image to array and add array
            test_photos_recto.append(img_to_array(img_rec))
            test_photos_verso.append(img_to_array(img_ver))
            # Add Image's label
            test_labels.append(label)
        index = index + 1

train_photos_recto = np.array(train_photos_recto)
train_photos_verso = np.array(train_photos_verso)
train_labels = np.array(train_labels)
test_photos_recto = np.array(test_photos_recto)
test_photos_verso = np.array(test_photos_verso)
test_labels = np.array(test_labels)

# Define the process for augmentation data
data_augmentation_r = tf.keras.Sequential([
     tf.keras.Input((img_height, img_width, 3)),
     layers.experimental.preprocessing.Rescaling(1./255),
     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
     layers.experimental.preprocessing.RandomRotation(0.2)])

data_augmentation_v = tf.keras.Sequential([
     tf.keras.Input((img_height, img_width, 3)),
     layers.experimental.preprocessing.Rescaling(1./255),
     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
     layers.experimental.preprocessing.RandomRotation(0.2)])

# Init the VGG model
vgg_conv_r = vgg16.VGG16(weights='imagenet', include_top=False,
                         input_shape=(img_height, img_width, 3))
vgg_conv_r._name = "vgg16_r"

vgg_conv_v = vgg16.VGG16(weights='imagenet', include_top=False,
                         input_shape=(img_height, img_width, 3))
vgg_conv_v._name = "vgg16_v"


# Layer of vgg isn't trainable
# Change name to can use twice the same model
for layer in vgg_conv_r.layers[:]:
    layer.trainable = False
    layer._name = layer._name  + str("_2")

for layer in vgg_conv_v.layers[:]:
    layer.trainable = False

# Define the network
# create parallel models
model_verso = data_augmentation_v.output
model_verso = vgg_conv_v(model_verso)
model_verso = layers.Flatten()(model_verso)

model_recto = data_augmentation_r.output
model_recto = vgg_conv_r(model_recto)
model_recto = layers.Flatten()(model_recto)

concat = layers.concatenate([model_recto,model_verso])

num_classes = 5

model_final = layers.Dense(4096*2, activation='relu')(concat)
model_final = layers.Dropout(0.5)(model_final)
model_final = layers.Dense(num_classes, activation='softmax')(model_final)


model = tf.keras.Model(inputs=[data_augmentation_r.input,
                               data_augmentation_v.input],
                       outputs=model_final)

#Callback
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='/home/genouest/inra_umr1349/mpalerme/model_multi_view_best.hd5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Compile the Network
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'],
  callbacks=[model_checkpoint_callback])

print(model.summary())


# Training the network
model.fit(
  x= [train_photos_recto, train_photos_verso], y=train_labels,
  validation_data = ([test_photos_recto, test_photos_verso], test_labels),
  epochs=150,
  verbose=2
)

# Save the network
model.save("/home/genouest/inra_umr1349/mpalerme/model_multi_view_last.hd5")

