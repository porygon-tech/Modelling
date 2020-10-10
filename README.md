LABEL DATA LOAD: <br>
Loads data.csv from given path (filepath variable) <br>
vector X: image id numbers <br>
vector y: text to be inferred from images <br>

IMAGE DATA TENSOR GENERATION: <br>
generates a 2000 x 128 x 128 x 3 array: <br>
  2000 images <br>
  128 x 128 pixels <br>
  3 colour channels (RGB) <br>
and stores it in imageSet.npy file (which can be later loaded using the np.load() function) <br>

TEXT-TO-NUMBER CATEGORY ID TRANSLATION: <br>
translates text stored in vector y to numbers <br>

NETWORK STRUCTURE GENERATION: <br>
builds the convolutional neural network model <br>
