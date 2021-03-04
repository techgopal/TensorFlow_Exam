import tensorflow as tf
from sentiment_analysis_4 import data_processing,model
import matplotlib.pyplot as plt
def train_model(file_name):
    #GET Data
    train,val,test = data_processing.get_data()

    AUTOTUNE = tf.data.AUTOTUNE

    train = train.cache().prefetch(buffer_size=AUTOTUNE)
    val = val.cache().prefetch(buffer_size=AUTOTUNE)
    test = test.cache().prefetch(buffer_size=AUTOTUNE)


    #Get Model
    m = model.get_model()
    m.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
    #Callback
    es_callback = tf.keras.callbacks.EarlyStopping(verbose=2)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('model_checkpoint/'+file_name,verbose=2,save_weights_only=True)
    history = m.fit(train,epochs=10,validation_data=val,callbacks=[es_callback,checkpoint_callback])
    m.save('trained_model/'+file_name)
    # Evaluate
    test_loss, test_accuracy = m.evaluate(test)
    print("Loss -> {}".format(test_loss))
    print("Acuuracy -> {}".format(test_accuracy))
    #Parameter
    h_parameters = history.history
    val_loss = h_parameters['val_loss']
    loss = h_parameters['loss']
    acc = h_parameters['binary_accuracy']
    val_acc = h_parameters['val_binary_accuracy']

    epochs = range(1, len(loss) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

