import tensorflow_datasets as tfds


def get_data():
    # Data From TF Datasets
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True)

    return train_data, validation_data, test_data
