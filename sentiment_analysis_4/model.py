import tensorflow as tf


def get_model():
    embedding_dim = 16
    max_features = 10000
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_features + 1, embedding_dim),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAvgPool1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.summary()
    return model
