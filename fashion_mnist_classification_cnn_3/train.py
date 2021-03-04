import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from fashion_mnist_classification_cnn_3 import data_processing, model as m


def train_model(file_name):


    print("Get Data")
    (x_train, y_train), (x_test, y_test) = data_processing.get_data()
    print(x_train.shape)
    print("Get Model")
    model = m.get_model()
    print("Compile Model")
    cp_callback = tf.keras.callbacks.ModelCheckpoint("model_checkpoint/" + file_name, verbose=2, save_weights_only=True)
    es_callback = tf.keras.callbacks.EarlyStopping(verbose=2)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics='accuracy')
    model.summary()
    print("Train Model")
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[cp_callback, es_callback])
    model.save("trained_model/" + file_name)
    print("Predictions ")
    y_pred = model.predict(x_test)
    print(np.argmax(y_pred[10]))
    print(y_test[10])
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_test[i])
        plt.xlabel(np.argmax(y_pred[i]))
        plt.ylabel(y_test[i])
    plt.show()
