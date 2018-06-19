'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# for timestamp
import datetime, os

batch_size = 128
num_classes = 10
epochs = 20
# standard timestamp
timestamp = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

# if a model name already not passed the code should assume you are trying to build a new model
# else it should look for and load the old model for which the name has been passed
# from the model folder
models_store = "models" + os.sep
model_name = "Model_0.109827256982_0.9841_2017_09_11_11_53_30"


# model name not provided
if model_name == "":

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # save the model
    pickle_filename = models_store + "Model_" + str(score[0]) + "_" + str(score[1]) + "_" + timestamp
    model_json = model.to_json()
    with open(pickle_filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(pickle_filename + ".h5")
    print("Saved model to disk")


# model name provided
else:
    # prediction run with the model
    # load model if needed
    pickle_filename = models_store + model_name
    # load json and create model
    json_file = open(pickle_filename + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(pickle_filename + ".h5")
    print("Model loaded from disk")

# test with my image
# import required package
from matplotlib import pyplot as plt
from random import randint

import numpy as np
from PIL import Image,ImageOps


im = Image.open('test.png')

im = im.convert('L') # grayscale
im = ImageOps.invert(im) # invert color
im = im.convert('1')  # make it pure black and white
pixels = np.asarray(im.getdata(), dtype=np.float64).reshape((im.size[1], im.size[0]))
print(type(pixels))
pixels = pixels.flatten() # flatten all the list of lists of row x col to a single array
# as 255 is the max number of color a pixel can have we device each element of the array
# to normailze the values of each pixel between 0 and 1
pixels = pixels / 255 # or max(pixels)
pixels = pixels.reshape(1, 784)
#print(pixels)

classification = model.predict(pixels)
print('NN predicted', np.argmax(classification, 1))
plt.imshow(pixels.reshape(28, 28), cmap=plt.cm.binary)
plt.show()

# # test with my data
# num = randint(0, x_test.shape[0])
# img = x_test[num]
# img = img.reshape(1, 784)
# print(type(img))
# print(img)
#
# classification = model.predict(img)
# #print(classification.reshape(1, 10))
# print('NN predicted', np.argmax(classification, 1))
# plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
# plt.show()
