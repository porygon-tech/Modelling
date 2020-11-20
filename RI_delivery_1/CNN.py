#==================
#=== TEST ZONE ====
#==================
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

#=== FEATURES FORMATTING ===
imageSet = []
for imgID in imgIDs:
	filename = str(filepath + "train_image_" + imgID.replace('\"', '') + ".png")
	imgPIL = tf.keras.preprocessing.image.load_img(filename, color_mode="grayscale")
	arr = tf.keras.preprocessing.image.img_to_array(imgPIL)
	imageSet.append([[arr]])

tx = np.array(imageSet)[:,0,0,:,:,:]

#=== MODEL BUILDING ===
num_train, img_rows, img_cols, img_channels =  tx.shape
num_classes = len(np.unique(ty))


checkpoint_path = csvpath + "guardado.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)


model = Sequential()
model.add(Conv2D(filters=img_rows, kernel_size=(3, 3), padding='same', activation='relu', input_shape = [img_rows, img_cols, img_channels]))

model.add(Conv2D(filters=img_rows, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(units = 128, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
history = model.fit(tx,ty, batch_size=10, epochs=10, verbose=1, callbacks=[cp_callback])

