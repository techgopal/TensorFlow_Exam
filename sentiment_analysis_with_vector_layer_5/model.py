import tensorflow as tf


def get_model(vectorized_layer,embedding_dim,max_features):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorized_layer,
        tf.keras.layers.Embedding(max_features + 1, embedding_dim),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAvgPool1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
    ])
    model.summary()
    return model
