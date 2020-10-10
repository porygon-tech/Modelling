#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np

print(tf.__version__)
img_height = 128
img_width = 128
#import PIL
#import PIL.Image
#(PIL.Image.open(r'/mnt/c/Users/Miki/Documents/Modelling/R&I/images_train/train_image_00891.png')).show()

#=== LABEL DATA LOAD ===
filepath = "/mnt/c/Users/Miki/Documents/Modelling/R&I/images_train/"
rawdata = np.genfromtxt(filepath + 'data.csv', dtype='<U8', delimiter=',')
print('data loaded')
y = rawdata[1:,1]
X = rawdata[1:,0]


#=== IMAGE DATA TENSOR GENERATION ===
#img_dir_list = []
imageSet = []
for imgID in X:
	print("Appending " + filepath + "train_image_" + imgID.replace('\"', '') + ".png")
	#img_dir_list.append(str(filepath + "train_image_" + imgID.replace('\"', '') + ".png"))
	imgPIL = tf.keras.preprocessing.image.load_img(filepath + "train_image_" + imgID.replace('\"', '') + ".png")
	arr = tf.keras.preprocessing.image.img_to_array(imgPIL)
	arr = np.array([arr])
	imageSet.append([arr])

imageSet = np.array([imageSet])
imageSet.shape
imageSet = imageSet[0,:,0,0,:,:]
imageSet.shape

np.save(filepath + "imageSet", imageSet)
#imageSet = np.load("imageSet.npy")


#=== TEXT-TO-NUMBER CATEGORY ID TRANSLATION ===
dictID = {}
IDcounter = 0
y_num = []
for textoutput in y:
	if textoutput not in dictID:
		dictID[textoutput] = IDcounter
		#print(textoutput, IDcounter)
		IDcounter += 1
	y_num.append(dictID[textoutput])

#len(dictID)



#=== NETWORK STRUCTURE GENERATION ===
CNN = Sequential()

CNN.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(128,128,3)))
CNN.add(Activation('relu'))
CNN.add(MaxPooling2D(pool_size=(2,2)))

CNN.add(Conv2D(filters=64, kernel_size=(3,3)))
CNN.add(Activation('relu'))
CNN.add(MaxPooling2D(pool_size=(2,2)))

CNN.add(Flatten())
CNN.add(Dense(units=64))

CNN.add(Dense(units=1, activation('softmax'))

CNN.summary()

CNN.compile(optimizer='adam', 
			 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			 metrics=['accuracy'])

CNN.fit(imageSet, y_num, epochs=500)

#INCLUDE CALLBACKS!!!





