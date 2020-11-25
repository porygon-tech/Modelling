import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import math

#https://www.codingforentrepreneurs.com/blog/install-tensorflow-gpu-windows-cuda-cudnn/
#=== FUNCTIONS ====
def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def splitdata(a, b, fraction):
	assert len(a) == len(b)
	length = len(a)
	train_num = math.floor(length * (1-fraction))
	return(a[0:train_num],b[0:train_num],a[train_num:],b[train_num:])

def showdata(mat, index):
	plt.imshow(mat[index,:,:,0].astype('float32'), interpolation='none', cmap=plt.cm.gray)
	plt.show()


#===============
#=== SCRIPT ====
#===============
csvpath = '/home/mroman/Master/images_train/'
filepath = csvpath + 'images/'

df = np.genfromtxt(csvpath + 'data.csv', dtype='<U8', delimiter=',')
labels = df[1:,1]
imgIDs = df[1:,0]


#=== LABEL FORMATTING ===
dictID = {}
IDcounter = 0
y_num = []
for textoutput in labels:
	if textoutput not in dictID:
		dictID[textoutput] = IDcounter
		#print(textoutput, IDcounter)
		IDcounter += 1
	y_num.append(dictID[textoutput])

ty = np.array(y_num)[:,np.newaxis]

num_classes = len(dictID)


#=== FEATURES FORMATTING ===
imageSet = []
for imgID in imgIDs:
	filename = str(filepath + "train_image_" + imgID.replace('\"', '') + ".png")
	imgPIL = tf.keras.preprocessing.image.load_img(filename, color_mode="grayscale")
	arr = tf.keras.preprocessing.image.img_to_array(imgPIL)
	imageSet.append([[arr]])

tx = (np.array(imageSet)[:,0,0,:,:,:] / 255).astype('float16')





#=== DATA SPLIT ===
'''
a: data   for training
b: labels for training
c: data   for testing
d: labels for testing
'''
randtx, randty = unison_shuffle(tx,ty)
a,b,c,d = splitdata(randtx,randty,0.2)
'''
np.save(filepath + "a", a)
np.save(filepath + "b", b)
np.save(filepath + "c", c)
np.save(filepath + "d", d)
'''

a = np.load(filepath + "a.npy")
b = np.load(filepath + "b.npy")
c = np.load(filepath + "c.npy")
d = np.load(filepath + "d.npy")

#=== MODEL BUILDING ===
num_train, img_rows, img_cols, img_channels =  tx.shape

model = Sequential()
model.add(Conv2D(filters=img_rows, kernel_size=(1, 3), padding='same', activation='relu', input_shape = [img_rows, img_cols, img_channels]))
model.add(Conv2D(filters=img_rows, kernel_size=(3, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units = 128, activation='relu'))
model.add(Dense(units = 128, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])







model.evaluate(c, d, batch_size=32)

#model.load_weights(checkpoint_path)


checkpoint_path = csvpath + "haha_1.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

model.fit(a, b, batch_size=32, epochs=10, verbose=1, callbacks=[cp_callback])

#history = model.fit(tx,ty, batch_size=32, epochs=10, verbose=1, callbacks=[cp_callback])

model.evaluate(c, d, batch_size=32)





for i in dictID.keys():
	print(i, dictID[i])


p = model.predict(c)

plt.imshow(p, interpolation='none', cmap='inferno')
plt.show()







checkpoint_path = csvpath + "fulltrained.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
model.fit(tx,ty, batch_size=32, epochs=10, verbose=1, callbacks=[cp_callback])









#=== PLAYGROUND ===


def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

randtx, randty = unison_shuffle(tx,ty)

import math
def splitdata(a, b, fraction):
	assert len(a) == len(b)
	length = len(a)
	train_num = math.floor(length * (1-fraction))
	return(a[0:train_num],b[0:train_num],a[train_num:],b[train_num:])

a,b,c,d = splitdata(tx,ty,0.2)



showdata(tx,5)
ty[5,0]
dictID['"eë"']

