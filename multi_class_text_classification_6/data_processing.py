import tensorflow as tf


def get_data():
    # url = 'http://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
    # dataset = tf.keras.utils.get_file("stack_overflow_16k.tar.gz",url,untar=True,cache_dir='.')

    train_ds = tf.keras.preprocessing.text_dataset_from_directory('datasets/train', batch_size=32, seed=32,
                                                                  validation_split=0.2, subset="training")
    val_ds = tf.keras.preprocessing.text_dataset_from_directory('datasets/train', batch_size=32, seed=32,
                                                                validation_split=0.2, subset="validation")
    test_ds = tf.keras.preprocessing.text_dataset_from_directory('datasets/test', batch_size=32)

    return train_ds, val_ds, test_ds
