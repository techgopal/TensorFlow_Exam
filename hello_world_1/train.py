import tensorflow as tf
from hello_world_1 import data_processing, model as m


def train_model(file_name):
    print(tf.version)
    print("Get Data")
    (x_train, y_train), (x_test, y_test) = data_processing.get_data()
    print("Get Model")
    model = m.get_model()
    print("Compile Model")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="model_checkpoint/" + file_name,
                                                     save_weights_only=True,
                                                     verbose=1)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy']
                  )
    model.summary()
    print("Train Model")
    model.fit(x_train, y_train, epochs=5, callbacks=[cp_callback])
    model.save("trained_model/" + file_name)
