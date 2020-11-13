#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np

#print(tf.__version__)

#=== PARAMETERS ===
batch_size = 32
img_height = 128
img_width = 128


#=== LABEL DATA LOAD ===
filepath = '/home/mroman/Master/images_train/'
rawdata = np.genfromtxt(filepath + 'data.csv', dtype='<U8', delimiter=',')
#print('data loaded')
y = rawdata[1:,1]
X = rawdata[1:,0]

#=== CATEGORY MAPPING DICTIONARY GENERATION ===
dictID = {}
IDcounter = 0
y_num = []
for textoutput in y:
	if textoutput not in dictID:
		dictID[textoutput] = IDcounter
		#print(textoutput, IDcounter)
		IDcounter += 1
	y_num.append(dictID[textoutput])

#=== IMAGE LOAD ===
import pathlib
data_dir = pathlib.Path(filepath)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  directory = filepath,
  labels = y_num,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#=== NETWORK STRUCTURE GENERATION ===
CNN = Sequential()

CNN.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(128,128,3)))
CNN.add(Activation('relu'))
CNN.add(MaxPooling2D(pool_size=(2,2)))

CNN.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(128,128,3)))
CNN.add(Activation('relu'))
CNN.add(MaxPooling2D(pool_size=(2,2)))

CNN.add(Flatten())
CNN.add(Dense(units=64))

CNN.add(Dense(units=1, activation('softmax')))

CNN.summary()

CNN.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
 

CNN.fit(train_ds, epochs=10)
CNN.save(
    filepath, overwrite=True, include_optimizer=True, save_format=None,
    signatures=None, options=None
)












