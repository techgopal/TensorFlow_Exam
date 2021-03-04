import tensorflow as tf
import re
import string

# Define
max_features = 10000
sequence_length = 250

def get_data():
    # Read Data
    train_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train', batch_size=32, seed=32,
                                                                  validation_split=0.2, subset='training')
    val_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train', batch_size=32, seed=32,
                                                                validation_split=0.2, subset='validation')
    test_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train', batch_size=32)

    return train_ds, val_ds, test_ds


def get_standardize(input_data):
    lower_case = tf.strings.lower(input_data)
    cleaned_data = tf.strings.regex_replace(lower_case, '<br />', '')
    regex_data = tf.strings.regex_replace(cleaned_data, '[%s]' % re.escape(string.punctuation), '')
    return regex_data

