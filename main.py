# from hello_world_1 import train,test
# from fashion_mnist_classification_2 import train
# from fashion_mnist_classification_cnn_3 import train
# from sentiment_analysis_4 import train
# from sentiment_analysis_with_vector_layer_5 import train
# from multi_class_text_classification_6 import  train
# from  predict_fuel_efficiency_8 import train
from keras_auto_tuner_9 import train
import tensorflow as tf

if __name__ == '__main__':
    print(tf.version)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.
    # VirtualDeviceConfiguration(memory_limit=2048)])
    ##train.train_model("getting_started")
    ##test.test_model()
    train.train_model("keras_auto_tuner_9")
