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

    #vectorized_layer = get_vectorized_layer()
    v_l = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=get_standardize,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    train_text = train_ds.map(lambda x, y: x)
    v_l.adapt(train_text)

    def get_vectorized_text(text, label):
        text = tf.expand_dims(text, -1)
        return v_l(text), label

    #Test
    text_batch, label_batch = next(iter(train_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print("Review", first_review)
    print("Label", train_ds.class_names[first_label])
    print("Vectorized review", get_vectorized_text(first_review, first_label))

    # # Vectorized
    final_train = train_ds.map(get_vectorized_text)
    final_val = val_ds.map(get_vectorized_text)
    final_test = test_ds.map(get_vectorized_text)

    return final_train, final_val, final_test


def get_standardize(input_data):
    lower_case = tf.strings.lower(input_data)
    cleaned_data = tf.strings.regex_replace(lower_case, '<br />', '')
    regex_data = tf.strings.regex_replace(cleaned_data, '[%s]' % re.escape(string.punctuation), '')
    return regex_data


# def get_vectorized_layer():
#     return tf.keras.layers.experimental.preprocessing.TextVectorization(
#         standardize=get_standardize(),
#         max_tokens=max_features,
#         output_mode='int',
#         output_sequence_length=500)


# def get_vectorized_text(text, label):
#     text = tf.expand_dims(text, -1)
#     vectorized_layer = get_vectorized_layer()
#     return vectorized_layer(text), label
