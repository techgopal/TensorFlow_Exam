import tensorflow as tf
import matplotlib.pyplot as plt
import string
from multi_class_text_classification_6 import data_processing, model
import re


def train_model(file_name):
    # 1. Read Data
    train_ds, val_ds, test_ds = data_processing.get_data()
    # Vectorization
    num_features = 10000
    num_embedings = 100
    seq_len = 250

    def text_standardize(input_text):
        lower_case = tf.strings.lower(input_text)
        remove_br = tf.strings.regex_replace(lower_case, '<br />', ' ')
        clean_text = tf.strings.regex_replace(remove_br, '[%s]' % re.escape(string.punctuation), ' ')
        return clean_text

    vectorized_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=text_standardize,
                                                                                    output_mode='int',
                                                                                    max_tokens=num_features,
                                                                                    output_sequence_length=seq_len
                                                                                    )
    # Execute Adapt only on Train Data
    only_text = train_ds.map(lambda x, y: x)
    vectorized_layer.adapt(only_text)
    # Reshape Data
    # f_train = train_ds.map(lambda x, l: (x, tf.keras.utils.to_categorical(l, num_classes=4)))
    # f_val = val_ds.map(lambda x, l: (x, tf.keras.utils.to_categorical(l, num_classes=4)))
    # f_test = test_ds.map(lambda x, l: (tf.expand_dims(x, -1), l))

    # Get Model
    m = model.get_model(vectorized_layer, num_features, num_embedings)
    # m.summary()
    m.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics='accuracy')

    # Train Model
    es_callback = tf.keras.callbacks.EarlyStopping()
    m.fit(train_ds, validation_data=val_ds, epochs=10,callbacks=[es_callback],validation_split=)
    m.save('trained_model/'+file_name)

    # Evaluate
    # text_batch, label_batch = next(iter(test_ds))
    # first_review, first_label = text_batch[0], label_batch[0]
    # print("Review", first_review)
    # print("Label", test_ds.class_names[first_label])
    # examples = [
    #     "The movie was great!",
    #     "The movie was okay.",
    #     "The movie was terrible..."
    # ]
    #
    # print("P Label", m.predict(first_review._numpy()))