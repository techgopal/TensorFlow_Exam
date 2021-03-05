import tensorflow as tf

from hello_world_1 import data_processing


def test_model():
    (x_train, y_train), (x_test, y_test) = data_processing.get_data()
    model = tf.keras.models.load_model('trained_model/getting_started')
    model.evaluate(x_test,  y_test, verbose=2)

test_model()