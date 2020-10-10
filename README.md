=== LABEL DATA LOAD ===
Loads data.csv from given path (filepath variable)
vector X: image id numbers
vector y: text to be inferred from images

=== IMAGE DATA TENSOR GENERATION ===
generates a 2000 x 128 x 128 x 3 array:
  2000 images
  128 x 128 pixels
  3 colour channels (RGB)
and stores it in imageSet.npy file (which can be later loaded using the np.load() function)

=== TEXT-TO-NUMBER CATEGORY ID TRANSLATION ===
translates text stored in vector y to numbers

=== NETWORK STRUCTURE GENERATION ===
builds the convolutional neural network model
