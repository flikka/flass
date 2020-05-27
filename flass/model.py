import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.color import gray2rgb
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logger = logging.getLogger()


class ImageFlattener(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n_samples = len(X)
        return X.reshape((n_samples, -1))


def svm_model():
    model = svm.SVC(probability=True)
    model.predict = model.predict_proba
    return model


def conv_model():
    model = tf.keras.models.Sequential()

    # Convolutional layer with 32 filters. Because 32 is best. Always.
    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),  # Filter size
            strides=(1, 1),
            padding="valid",
            activation="relu",
            input_shape=(28, 28, 3),  # All important images are 28 pixels
        )
    )

    # Max pooling, all the cool people do it.
    model.add(
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),  # Size feature will be mapped to
            strides=(2, 2),  # How the pool "steps" across the feature
        )
    )

    # Dropout, of course
    model.add(tf.keras.layers.Dropout(rate=0.25))  # Randomly disable 25% of neurons

    # Make 2d output 1d to feed into the classification section
    model.add(tf.keras.layers.Flatten())

    # Final layer, outputting 10 probabilities. Cause all good problems have 10 classes
    model.add(
        tf.keras.layers.Dense(
            units=10,  # Output shape
            activation="softmax",  # Softmax Activation Function
        )
    )

    # Build the model
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    return model


def get_data(data_key, subset=-1):
    logger.info(f"Using {data_key} dataset, reducing training set to {subset} elements")
    if data_key == "fashion":
        data_classes = get_fashion_data()
    elif data_key == "mnist":
        data_classes = get_mnist_data()
    else:
        raise ValueError(f"Unsupported value for 'data_key': {data_key}")

    ((x_train, y_train), (x_test, y_test)), class_names = data_classes

    x_train = np.array([gray2rgb(grayimg) for grayimg in x_train])
    x_test = np.array([gray2rgb(grayimg) for grayimg in x_test])

    if subset != -1:
        x_train = x_train[:subset]
        y_train = y_train[:subset]

    data_classes = ((x_train, y_train), (x_test, y_test)), class_names
    return data_classes


def get_mnist_data():
    logger.info("Downloading MNIST dataset")
    data = tf.keras.datasets.mnist.load_data()
    logger.info("Done downloading MNIST dataset")
    # Map for human readable class names
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return data, class_names


def get_fashion_data():
    logger.info("Downloading Fashion dataset")
    data = tf.keras.datasets.fashion_mnist.load_data()
    logger.info("Done downloading Fashion dataset")
    # Map for human readable class names
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    return data, class_names


def train(x, y, batch_size, epochs, model_type):
    logger.info(f"Starting to build {model_type} model")
    if model_type == "kerasconv":
        model = conv_model()
        model.summary()
        pipeline_steps = [("model", model)]
        full_pipeline = Pipeline(steps=pipeline_steps)
        full_pipeline.fit(x, y, model__batch_size=batch_size, model__epochs=epochs)
        return full_pipeline

    elif model_type == "svm":
        pipeline_steps = [("image_flattener", ImageFlattener())]
        model = svm_model()
        pipeline_steps.append(("model", model))
        full_pipeline = Pipeline(steps=pipeline_steps)
        full_pipeline.fit(x, y)
        return full_pipeline


def plot_incorrect(x_test, y_test, y_predicted, class_names):
    incorrect = np.nonzero(y_predicted != y_test)[0]
    # Display the first 16 incorrectly classified images from the test data set
    plt.figure(figsize=(15, 8))

    for j, incorrect in enumerate(incorrect[0:8]):
        plt.subplot(2, 4, j + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_test[incorrect].reshape(28, 28), cmap="Reds")
        plt.title("Predicted: {}".format(class_names[y_predicted[incorrect]]))
        plt.xlabel("Actual: {}".format(class_names[y_test[incorrect]]))

    plt.show()
