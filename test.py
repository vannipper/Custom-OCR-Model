# Custom OCR Model by Van Nipper:
# This model is able to automatically read individual words,
# and excels at specifically reading my own handwriting.

import os
import gzip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def load_emnist_images(filename):
    """Load EMNIST images from the provided .gz file."""
    with gzip.open(filename, 'rb') as f:
        # Skip the first 16 bytes (header information)
        f.read(16)
        # Read the image data
        buf = f.read()
        # Convert the byte buffer to a NumPy array
        data = np.frombuffer(buf, dtype=np.uint8)
        # Reshape the data to (number of images, 28, 28)
        images = data.reshape(-1, 28, 28)
        return images

def load_emnist_labels(filename):
    """Load EMNIST labels from the provided .gz file."""
    with gzip.open(filename, 'rb') as f:
        # Skip the first 8 bytes (header information)
        f.read(8)
        # Read the label data
        buf = f.read()
        # Convert the byte buffer to a NumPy array
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

def loadEmnist():
    # File paths for the EMNIST dataset
    base_dir = "/Users/cnipper/tensorflow_datasets/emnist/byclass"
    test_images_path = os.path.join(base_dir, 'emnist-byclass-test-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(base_dir, 'emnist-byclass-test-labels-idx1-ubyte.gz')

    emnist_test = load_emnist_images(test_images_path) # load dataset
    emnist_test_labels = load_emnist_labels(test_labels_path)

    slice_index = int(-0.05 * len(load_emnist_images(test_images_path)))  # Get original dataset length, slice images and labels
    emnist_test = np.expand_dims(emnist_test[slice_index:], -1)
    emnist_test_labels = emnist_test_labels[slice_index:]

    return emnist_test, emnist_test_labels

def loadUser():
    image_paths = (line for line in open("data/user/user.csv").readlines()[1:])
    dataset = [entry.split(',') for entry in image_paths]

    # Create two lists
    image_paths, labels = [], []
    for image, label in dataset:
        image_paths.append(image)
        labels.extend([label.strip('\n')] * 8) # append 8 times

    # Preprocess each image in image_paths
    images = []
    for image_path in image_paths:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.resize(image, [28, 28])
        image = image / 255.0
        image = 1 - image  # Invert the pixel values
        for i in range(4):
            for j in range(2):
                images.append(tf.image.rot90(tf.image.flip_left_right(image) if j == 1 else image, i))

    # Convert labels (characters) to numeric values
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Convert to numpy arrays for shuffling
    images = np.array(images)
    labels_encoded = np.array(labels_encoded)

    # Shuffle images and labels with the same random permutation
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels_encoded = labels_encoded[indices]

    return images, labels_encoded

def viewImages(images, labels, num_images_to_display):
    # Display the first 5 images and labels in the test set
    plt.figure(figsize=(10, 5))

    for i in range(num_images_to_display):
        # Select an image and its label
        image = images[i].squeeze()  # Remove the last dimension if it's (28, 28, 1)
        label = labels[i]

        # Display the image
        plt.subplot(1, num_images_to_display, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')

    plt.show()

# SET UP DATA STRUCTURE
emnist_test, emnist_test_labels = loadEmnist()
# viewImages(emnist_test, emnist_test_labels, 5)

user_test, user_test_labels = loadUser()
# viewImages(user_test, user_test_labels, 5)

# RUN OCR MODELS ON DATA
model1name = input('Enter the name of model1: ')
model2name = input('Enter the name of model2: ')
model1 = tf.keras.models.load_model(f'models/{model1name}.keras')
model2 = tf.keras.models.load_model(f'models/{model2name}.keras')
print('Testing model 1: (trained without user data)')
print(f'Normal data: {model1.evaluate(emnist_test, emnist_test_labels)}\nUser data: {model1.evaluate(user_test, user_test_labels)}')
print('Tesitng model 2: (trained with user data)')
print(f'Normal data: {model2.evaluate(emnist_test, emnist_test_labels)}\nUser data: {model2.evaluate(user_test, user_test_labels)}')

# EXIT
i = input('Press enter to exit.')
os.system('clear') # clear screen