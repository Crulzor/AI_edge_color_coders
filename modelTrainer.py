#SOURCES: https://medium.com/@a.ayyuced/image-classification-models-on-arduino-nano-33-ble-sense-60bf845fd2aa
#great source for explaining the basics of 2D convolution as well

#first we import a whole bunch of stuff from the tf.keras library
import numpy as np
import tensorflow as tf
import random

assert tf.__version__.startswith('2')
from PIL import Image, ImageOps, ImageChops, ImageFilter
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from keras.preprocessing import image


import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#first we declare some global variables and the image path
#the batch size is necessary because we don't have the same amount of pics 
#in every class/folder
batch_size = 50
img_height = 96
img_width = 96

#declare image path. Classes are derived from subfolders
train_path = "./img"
test_path = "./img"
#load the data and split into test data, validation data and training data
#We also reduce the image size to reduce the model (hopefully)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_path, 
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_path,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)
#train the model with the image classifier (5 epochs)
#THIS DIDNT WORK, USES UP TOO MUCH MEMORY 
#we have to use a sequential model with 2D convolution


model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        Conv2D(16, 3, activation='relu', padding='SAME'),
        MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.DepthwiseConv2D(8, 3, activation='relu', padding='SAME'),
        MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.DepthwiseConv2D(8, 3, activation='relu', padding='SAME'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=4, activation='softmax'),
    ])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy'])
  
history = model.fit(train_ds, validation_data=val_ds, epochs=100, batch_size=batch_size)



x_test= tf.concat([x for x, y in test_ds], axis=0)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
images = tf.cast(x_test[0], tf.float32)/255.0
mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
def representative_data_gen():
  for input_value in test_ds.take(100):
    yield [input_value]

converter.representative_dataset = representative_data_gen
import sys
import os
if sys.version_info.major >= 3:
    import pathlib
else:
    import pathlib2 as pathlib

tflite_quant_model = converter.convert()
tflite_models_dir = pathlib.Path("./models")
tflite_model_quant_file = tflite_models_dir/"model.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)




plt.plot()