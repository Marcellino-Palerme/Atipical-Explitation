#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import layers
from keras.applications import vgg16, vgg19
from copy import *
import numpy as np

batch_size = 20
# Define size of image
img_height = 224
img_width = 224

# Define directory where take image
# In directoty there is one directory by class
data_dir = "./deep"
data_fit = data_dir + "/fit"

# Take images for training
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_fit,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='int')

# Take images for validation
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_fit,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='int')

# Show all classes
class_names = train_ds.class_names
print(class_names)


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



  # Define the process for augmentation data
data_augmentation = tf.keras.Sequential([
     layers.experimental.preprocessing.Rescaling(1./255),
     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
     layers.experimental.preprocessing.RandomRotation(0.2),
     layers.experimental.preprocessing.RandomZoom(0.2)])


# Augment the datas
train_aug_ds = train_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))
val_aug_ds = val_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))

"""
for i in range(9):
    temp = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y))
    train_aug_ds = train_aug_ds.concatenate(temp)
    temp = val_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y))
    val_aug_ds = val_aug_ds.concatenate(temp)
"""

# Init the VGG model
vgg_conv_r = vgg16.VGG16(weights='imagenet', include_top=False,
                         input_shape=(img_height, img_width, 3))

vgg_conv_v = vgg16.VGG16(weights='imagenet', include_top=False,
                         input_shape=(img_height, img_width, 3))

i = 0
# Training all the layers
for layer in vgg_conv_r.layers[:]:
    layer.trainable = True
    layer._name = layer._name  + str("_2")
    i = i+ 1

for layer in vgg_conv_v.layers[:]:
    layer.trainable = True

model_verso = vgg_conv_v.output
model_verso = layers.Flatten()(model_verso)

model_recto = vgg_conv_r.output
model_recto = layers.Flatten()(model_recto)

concat = layers.concatenate([model_recto,model_verso])

num_classes = len(class_names)

# Define the network
model_final = layers.Dense(4096*2, activation='relu')(concat)
model_final = layers.Dropout(0.5)(model_final)
model_final = layers.Dense(num_classes, activation='softmax')(model_final)


model = tf.keras.Model(inputs=[vgg_conv_r.input, vgg_conv_v.input],
                       outputs=model_final)

# Compile the Network
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

print(model.summary())

train_ds_1 = copy(train_ds)
val_ds_1 = copy(val_ds)

train = []
t_label = []
val = []
v_label = []

for value in train_aug_ds.take(-1):
  for index in range(len(value)):
    train.append(value[0][index])
    t_label.append(value[1][index])

for value in val_aug_ds.take(-1):
  for index in range(len(value)):
    val.append(value[0][index])
    v_label.append(value[1][index])

train = np.array(train)
t_label = np.array(t_label)
val = np.array(val)
v_label = np.array(v_label)

print(train[0])


# Training the network
model.fit(
  x= [train, train], y=t_label,
  validation_data=([val, val], v_label),
  epochs=1,
)

# Save the network
model.save("./model.hd5")

