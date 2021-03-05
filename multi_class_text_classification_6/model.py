import tensorflow as tf


def get_model(vectorized_layer, num_features, num_embedding):
    model = tf.keras.Sequential([
        vectorized_layer,
        tf.keras.layers.Embedding(num_features + 1, num_embedding),
        tf.keras.layers.GlobalAvgPool1D(),
        tf.keras.layers.Dense(16,activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    return model
