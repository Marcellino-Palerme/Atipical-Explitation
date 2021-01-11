#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg16


batch_size = 20
# Define size of image
img_height = 224
img_width = 224

# Define directory where take image
# In directoty there is one directory by class
data_dir = "/home/port-mpalerme/Documents/Atipical/Traitement/photos/deep"
data_fit = data_dir + "/fit"

# Take images for training
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_fit,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Take images for validation
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_fit,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Show all classes
class_names = train_ds.class_names
print(class_names)


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



  # Define the process for augmentation data
data_augmentation = tf.keras.Sequential([
     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
     layers.experimental.preprocessing.RandomRotation(0.2),
     layers.experimental.preprocessing.RandomZoom(0.2)])


# Augment the datas
train_aug_ds = train_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))
val_aug_ds = val_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))

for i in range(3):
    temp = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y))
    train_aug_ds = train_aug_ds.concatenate(temp)
    temp = val_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y))
    val_aug_ds = val_aug_ds.concatenate(temp)


# Init the VGG model 
vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False,
                       input_shape=(img_height, img_width, 3))

# Freeze all the layers
for layer in vgg_conv.layers[:]:
    layer.trainable = False

num_classes = len(class_names)

# Define the network
model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  vgg_conv,
  layers.Flatten(),
  layers.Dense(4096, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(num_classes, activation='softmax')
])

# Compile the Network
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# Training the network
model.fit(
  train_aug_ds,
  validation_data=val_aug_ds,
  epochs=50
)

# Save the network
model.save("/home/port-mpalerme/Documents/Atipical/Traitement/model.hd5")
