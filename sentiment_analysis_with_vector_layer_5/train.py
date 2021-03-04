import tensorflow as tf
from sentiment_analysis_with_vector_layer_5 import data_processing, model
import matplotlib.pyplot as plt
import re
import string


def get_standardize(input_data):
    lower_case = tf.strings.lower(input_data)
    cleaned_data = tf.strings.regex_replace(lower_case, '<br />', '')
    regex_data = tf.strings.regex_replace(cleaned_data, '[%s]' % re.escape(string.punctuation), '')
    return regex_data


def train_model(file_name):
    # GET Data
    train, val, test = data_processing.get_data()

    max_features = 10000
    sequence_length = 250
    # vectorized_layer
    vectorized_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=get_standardize,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    train_text = train.map(lambda x, y: x)
    vectorized_layer.adapt(train_text)

    final_train = train.map(lambda x, y: (tf.expand_dims(x, -1), y))
    final_val = val.map(lambda x, y: (tf.expand_dims(x, -1), y))
    final_test = test.map(lambda x, y: (tf.expand_dims(x, -1), y))

    AUTOTUNE = tf.data.AUTOTUNE

    train = final_train.cache().prefetch(buffer_size=AUTOTUNE)
    val = final_val.cache().prefetch(buffer_size=AUTOTUNE)
    test = final_test.cache().prefetch(buffer_size=AUTOTUNE)

    # Get Model
    m = model.get_model(vectorized_layer,embedding_dim=50,max_features=max_features)
    m.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
    # Callback
    es_callback = tf.keras.callbacks.EarlyStopping(verbose=2)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('model_checkpoint/' + file_name, verbose=2,
                                                             save_weights_only=True)
    history = m.fit(train, epochs=10, validation_data=val, callbacks=[es_callback, checkpoint_callback])
    # m.save('trained_model/'+file_name)
    # # Evaluate
    # test_loss, test_accuracy = m.evaluate(test)
    # print("Loss -> {}".format(test_loss))
    # print("Acuuracy -> {}".format(test_accuracy))
    # #Parameter
    # h_parameters = history.history
    # val_loss = h_parameters['val_loss']
    # loss = h_parameters['loss']
    # acc = h_parameters['binary_accuracy']
    # val_acc = h_parameters['val_binary_accuracy']
    #
    # epochs = range(1, len(loss) + 1)
    #
    # # "bo" is for "blue dot"
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # # b is for "solid blue line"
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # plt.show()
