from keras.models import Sequential
from keras.layers import Dense
from mnist import MNIST
import os
import numpy as np
import keras.utils.np_utils as util
# -----------------------------------------------------------------------------
# The current working directory
wk_dir = os.path.abspath('..')
# -----------------------------------------------------------------------------
# Load the MNIST dataset
mndata = MNIST(wk_dir + '/Data')
# -----------------------------------------------------------------------------
# Get the traininng data and labels
train_image, train_label = mndata.load_training()
# Convert them to numpy array
train_image = np.asarray(train_image)
train_label = np.asarray(train_label)
# Convert label to a vector of ten elements
train_label = util.to_categorical(train_label)
# -----------------------------------------------------------------------------
# Get the testing data and labels
test_image, test_label = mndata.load_testing()
# Convert them to numpy array
test_image = np.asarray(test_image)
test_label = np.asarray(test_label)
# Convert the label to a vector of ten elements
test_label = util.to_categorical(test_label)
# -----------------------------------------------------------------------------
# Add two layers. The first layer has 784 units (input layer).
# The second layer has 10 units (hidden layer).
# The third layer has 10 units (output layer).
model = Sequential()
model.add(Dense(units=10, activation='sigmoid', input_dim=784))
model.add(Dense(units=10, activation='sigmoid'))
# Compile. Cross entropy cost functions, stochastic gradient descent, accuracy 
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# -----------------------------------------------------------------------------
# Fit the model with batch_size = 32
model.fit(train_image, train_label, epochs=5, batch_size=32)
# Evaluate the performance of the model
loss = model.evaluate(test_image, test_label, batch_size=128)
# Print the accuracy of this model
print("\n\nAccuracy: " , loss[0])
