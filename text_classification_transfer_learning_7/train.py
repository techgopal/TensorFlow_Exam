from text_classification_transfer_learning_7 import model, data_processing

import tensorflow as tf


def train_model(file_name):
    # 1. Import Data
    train, val, test = data_processing.get_data()

    # 2. Import model
    m = model.get_model()
    # m.summary()

    m.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
