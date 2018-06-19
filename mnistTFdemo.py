# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

from matplotlib import pyplot as plt
from random import randint

import numpy as np
import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

  # # test with my data
  # num = randint(0, mnist.test.images.shape[0])
  # img = mnist.test.images[num]
  # print(type(img))
  # print(img)
  #
  # classification = sess.run(tf.argmax(y, 1), feed_dict={x: [img]})
  # #plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
  # #plt.show()
  # print('NN predicted', classification[0])

  # test with my image
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
  print(pixels)

  classification = sess.run(tf.argmax(y, 1), feed_dict={x: [pixels]})
  plt.imshow(pixels.reshape(28, 28), cmap=plt.cm.binary)
  print('NN predicted:---------->', classification)
  plt.show()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
