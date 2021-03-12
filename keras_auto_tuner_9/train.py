import tensorflow as tf
from keras_auto_tuner_9 import model, data_processing
import kerastuner as kt


def train_model(file_name):
    (X_train, y_train), (X_test, y_test) = data_processing.get_data()

    # 2. Build Model
    tuner = kt.Hyperband(model.build_model, objective='val_accuracy', max_epochs=15,
                         factor=3, directory="keras_autotuner/" + file_name,
                         project_name='intro_to_kt')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_model = tuner.hypermodel.build(best_hps)

    history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[stop_early])

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    # Train Again
    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(X_train, y_train, epochs=best_epoch)

    eval_result = hypermodel.evaluate(X_test, y_test)
    print("[test loss, test accuracy]:", eval_result)

