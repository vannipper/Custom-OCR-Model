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
    train_images_path = os.path.join(base_dir, 'emnist-byclass-train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(base_dir, 'emnist-byclass-train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(base_dir, 'emnist-byclass-test-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(base_dir, 'emnist-byclass-test-labels-idx1-ubyte.gz')

    # Load the dataset
    emnist_train = load_emnist_images(train_images_path)
    emnist_train_labels = load_emnist_labels(train_labels_path)
    emnist_test = load_emnist_images(test_images_path)
    emnist_test_labels = load_emnist_labels(test_labels_path)

    fraction = 0.10
    num_train_samples = int(fraction * emnist_train.shape[0])
    num_test_samples = int(fraction * emnist_test.shape[0])

    # Slice the datasets
    emnist_train = emnist_train[:num_train_samples]
    emnist_train_labels = emnist_train_labels[:num_train_samples]
    emnist_test = emnist_test[:num_test_samples]
    emnist_test_labels = emnist_test_labels[:num_test_samples]

    # Expand dimensions to fit the (28, 28, 1) format required by CNN
    emnist_train = np.expand_dims(emnist_train, -1)
    emnist_test = np.expand_dims(emnist_test, -1)

    return emnist_train, emnist_train_labels, emnist_test, emnist_test_labels

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

    split_index = int(len(images) * 1)
    user_train = images[:split_index]
    user_test = images[split_index:]
    user_train_labels = labels_encoded[:split_index]
    user_test_labels = labels_encoded[split_index:]

    return user_train, user_train_labels, user_test, user_test_labels

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

def buildModel():
    # CNN
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))  # convolutional / pooling layers
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten()) # fully connected (dense) layers
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(62, activation='softmax'))  # 62 classes/outputs (0-9, A-Z, a-z))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # compile the model

    return model

emnist_train, emnist_train_labels, emnist_test, emnist_test_labels = loadEmnist()
# viewImages(emnist_train, emnist_train_labels, 5)

user_train, user_train_labels, user_test, user_test_labels = loadUser()
# viewImages(user_train, user_train_labels, 5)
 
# Combine training images and labels (if training user model)
emnist_train = np.concatenate((emnist_train, user_train), axis=0)
emnist_train_labels = np.concatenate((emnist_train_labels, user_train_labels), axis=0)
emnist_test = np.concatenate((emnist_test, user_test), axis=0)
emnist_test_labels = np.concatenate((emnist_test_labels, user_test_labels), axis=0)

# CNN
model = buildModel()
model.fit(emnist_train, emnist_train_labels, epochs=10, batch_size=32, validation_data=(emnist_test, emnist_test_labels))
modelname = input('Enter the name of your new model: ')
model.save(f'models/{modelname}.keras')
os.system('clear') # clear screen