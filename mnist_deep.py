''' A deep MNIST classifier using Convolutional layers '''

from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data #Loads MNIST data
#mnist is a lightweight class that stores the training, validation and testing, sets as NumPy arrays
#Also provides a function for iterating through data minibatches
mnist = input_data.read_data_sets('MNIST_data', one_hot = True) 

import tensorflow as tf
sess = tf.InteractiveSession() #this class makes tensorflow more flexible about hw the code is structured
							   #allows to interleave operations which build the graph with ones that run i



## BUILDING A SOFTMAX REGRESSION MODEL

# PLACEHOLDERS
# building computation graph by creating nodes for input images and target output classes
# Placeholders are a value that we will input when we ask tf to run a computation
# Input image x will consist a 2d tensor of floating point numbers, and we will assign it a shape of [None, 784]
# where 784 is the dimensionality of a single flattened 28 by 28 pixel MNIST image, None indicates that the first 
# dimension, corrosponding to the batch size, can be of any size

# The target output classes y_ will also consist a 2d tensor, each row is a one-hot 10-dim vector indicating which
# digit class (0-9) the corrosponding MNIST image belongs in 
# Shape argument is optional but helps TF to catch bugs stemming from inconsistent tensor shapes

x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])

# VARIABLES
# A value that lives in TF's computation graph. It can be used and modified by computation
# defining weights W and biases b for the model. These can be treated like additional inputs
# initailize both as tensors full of zeros
# W is a 784 x 10 matrix --> 784 input features and 10 outputs
# b is a 10d vector

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# Here we initialize the variables using this session
sess.run(tf.global_variables_initializer())


# PREDICTED CLASS AND LOSS FUNCTION
# Regression model can now be implemented in one line! We multiply the vectorized input images  by the weight W and add b

y = tf.matmul(x,W) + b

# Now to specify the loss function. Loss indicates how bad the model's prediction was on a single example
# The goal is to ofcourse minimize that while training across all the examples
# Loss function here is the cross entropy bw the target and the softmax activation function applied
# to the model's prediction
# tf.nn.softmax_cross_entropy_with_logits internally applies softmax on the model's unnormalized
# model prediction and sums across all classes, and tf.reduce_mean takes the average sums across all classes
# tf.reduce_mean takes the average over these sums

cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


## TRAIN THE MODEL

# TF will use automatic differentiation to find the gradients of the loss wrt each of the variables
# TF has a bunch of built in optimization algorithms to achieve this
# here I will use steepest gradient descent, w a step length of 0.5, to decend the cross entropy
# TF will add new operations to the computation through this one line
# these include ones to compute gradients, compute parameter update steps and apply update steps to parameters
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# when train_step is returned above, it can apply gradient decent updates to parameters. So the model can 
# be trained by repeatedly using train_step

for _ in range (1000):
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict = {x:batch[0], y_:batch[1]})

# Above I am loading 100 training examples in each training interation. I then run the training_step operation
# using feed_dict to replace the placeholder tensors x and y_ with training examples
# Note: Any tensor in the graph can be replaced using feed_dict - not just placeholders


## MODEL EVALUATION

# First things first, we will figure out where we predicted the correct label
# tf.argmax is useful as it gives the index of the highest entry in a tensor along some axis
# For example, tf.argmax(y,1) is the label our model thinks is most likely for each input
# while tf.argmax(y_,1) is the true label. We can use tf.equal to check if our
# prediction matches the truth.

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# this will give us a list of booleans, to determine the correct fraction we cast to 
# floating point numbers and then take the mean
# For instance: [True, False, True, True] would become [1,0,1,1] which would become 0.75

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# finally evaluating the accuracy on test data

print ('Well done, your accuracy is: ', accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))

