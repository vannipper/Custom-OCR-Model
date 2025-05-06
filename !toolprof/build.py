# VAN NIPPER - TOOL PROFICIENCY
# Using tutorial by Neural Nine - Python Handwritten Digit Recognition

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# LOAD DATA

mnist = tf.keras.datasets.mnist # data for handwritten digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1) # convert all data to scale of (0-1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# BUILD MODEL

model = tf.keras.models.Sequential() # use sequential NN model
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) # turns image into a one dimensional array
model.add(tf.keras.layers.Dense(128, activation='relu')) # select activation function
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) # output layer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# TRAIN MODEL

model.fit(x_train, y_train, epochs=3)
model.save('handwrittenmodel.keras')