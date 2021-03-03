import tensorflow as tf
from tensorflow.keras import backend as K


def get_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # Normalize values between 0 & 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape((x_train.shape[0], 1, 28, 28))
        x_test = x_test.reshape((x_test.shape[0], 1, 28, 28))
    else:
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    return (x_train, y_train), (x_test, y_test)
