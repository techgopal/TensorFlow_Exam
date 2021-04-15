import tensorflow_datasets as tfds
import tensorflow as tf
def get_data():
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    def normalized_data(input_image, input_mask):
        input_image = tf.cast(input_image,tf.float32) / 255.0
        input_mask = input_mask - 1
        return input_image,input_mask

    @tf.function
    def load_train(data_point):
        input_image = tf.image.resize(data_point['image'],(128,128))
        input_mask = tf.image.resize(data_point['segmentation_mask'],(128,128))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image,input_mask = normalized_data(input_image,input_mask)
        return input_image,input_mask

    @tf.function
    def load_test(data_point):
        input_image = tf.image.resize(data_point['image'], (128, 128))
        input_mask = tf.image.resize(data_point['segmentation_mask'], (128, 128))

        input_image, input_mask = normalized_data(input_image, input_mask)
        return input_image, input_mask

    train = dataset['train'].map(load_train, num_parallel_calls=tf.data.AUTOTUNE)
    test = dataset['test'].map(load_test)

    return train, test, info