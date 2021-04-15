from image_segmentation_10 import data_processing, model
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
def train_model(file_name):
    train,test, info = data_processing.get_data()

    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = 32
    BUFFER_SIZE = 50
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    EPOCHS = 2
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE)

    def display(display_list):
        plt.figure(figsize=(15, 15))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()

    for image, mask in train.take(1):
        sample_image, sample_mask = image, mask
    # display([sample_image, sample_mask])

    m = model.build_model()
    m.summary()

    def create_mask(pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]

    def show_predictions(dataset=None, num=1):
        if dataset:
            for image, mask in dataset.take(num):
                pred_mask = m.predict(image)
                display([image[0], mask[0], create_mask(pred_mask)])
        else:
            display([sample_image, sample_mask,
                     create_mask(m.predict(sample_image[tf.newaxis, ...]))])

    show_predictions()



    class DisplayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            clear_output(wait=True)
            show_predictions()
            print('\nSample Prediction after epoch {}\n'.format(epoch + 1))

    model_history = m.fit(train_dataset, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_dataset,
                              callbacks=[DisplayCallback()])
