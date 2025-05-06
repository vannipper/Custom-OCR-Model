import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# LOAD DATA

mnist = tf.keras.datasets.mnist # data for handwritten digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1) # convert all data to scale of (0-1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# LOAD MODEL

model = tf.keras.models.load_model('handwrittenmodel.keras')

# EVALUATE MODEL

loss, accuracy = model.evaluate(x_test, y_test)

image_number = 0
while os.path.isfile(f"/Users/cnipper/Desktop/Desktop Files/Fall 2024/Neural Networks/toolprofdata/{image_number}.png"):
    try:
        img = cv2.imread(f"/Users/cnipper/Desktop/Desktop Files/Fall 2024/Neural Networks/toolprofdata/{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}.")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print(f"Image {image_number}.png not found!")
    finally:
        image_number += 1