import tensorflow as tf


def get_model(normalization_layer):
    model = tf.keras.Sequential([
        normalization_layer,
        tf.keras.layers.Dense(1)
    ])

    return model
