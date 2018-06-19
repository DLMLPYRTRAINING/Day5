'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

# For hyper parameter optimization
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

# for timestamp
import datetime, os

batch_size = 128
num_classes = 10
# standard timestamp
timestamp = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

# if a model name already not passed the code should assume you are trying to build a new model
# else it should look for and load the old model for which the name has been passed
# from the model folder
models_store = "models" + os.sep
model_name = "Model_0.081652014855_0.984_2017_09_20_10_27_42" # "Model_0.0828318153799_0.9827_2017_09_11_13_12_02" # "Model_0.109827256982_0.9841_2017_09_11_11_53_30"


# Data preparation
def data():
    """
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes = 10
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test

# model preparation
def model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Dense(100))

        # We can also choose between complete sets of layers

        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=20,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



# model name not provided
if model_name == "":

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    score, acc = best_model.evaluate(X_test, Y_test)
    print("Evaluation of best performing model:")
    print(score, acc)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)



    # save the model
    pickle_filename = models_store + "Model_" + str(score) + "_" + str(acc) + "_" + timestamp
    model_json = best_model.to_json()
    with open(pickle_filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    best_model.save_weights(pickle_filename + ".h5")
    print("Saved model to disk")

    model = best_model

#--------------------------------------------------------------------------
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


#------------------------------------------------------------------
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
# print(type(pixels))
pixels = pixels.flatten() # flatten all the list of lists of row x col to a single array
# as 255 is the max number of color a pixel can have we device each element of the array
# to normailze the values of each pixel between 0 and 1
pixels = pixels / 255 # or max(pixels)
pixels = pixels.reshape(1, 784)
#print(pixels)


classification = model.predict(pixels)
print('NN predicted:---------->', np.argmax(classification, 1))
plt.imshow(pixels.reshape(28, 28), cmap=plt.cm.binary)
plt.show()
