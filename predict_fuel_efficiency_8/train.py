from predict_fuel_efficiency_8 import model, data_processing

import tensorflow as tf
import numpy as np

def train_model(file_name):
    # 1. Import Data
    X_train, y_train, X_test, y_test = data_processing.get_data()
    
    X_train.head()

    # Normaliztion Layer
    normalization_layer = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[1, ])
    normalization_layer.adapt(np.array(X_train))

    # 2. Import model
    m = model.get_model(normalization_layer)
    m.summary()
    # 3. Compile
    m.compile(optimizer='adam', loss='mean_absolute_error')
    # 4. Fit
    history = m.fit(X_train.values, y_train.values, epochs=100, verbose=1, validation_split=0.2)
    #5. Evaluate
    m.evaluate(X_test.values, y_test.values, verbose=0)