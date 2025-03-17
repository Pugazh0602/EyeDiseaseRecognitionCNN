# data_preprocessing.py
import tensorflow as tf
from tensorflow.keras import layers

def preprocess_data(data, augmentation=True):
    """
    Preprocesses the data by normalizing pixel values and optionally applying data augmentation.

    Args:
        data (tf.data.Dataset): The TensorFlow Dataset to preprocess.
        augmentation (bool): Whether to apply data augmentation.

    Returns:
        tuple: A tuple containing the train, validation, and test datasets.
    """

    # Normalize pixel values to [0, 1]
    data = data.map(lambda x, y: (x / 255, y))

    # Split data into train, validation, and test sets
    train_size = int(0.7 * len(data)) + 1
    val_size = int(0.2 * len(data))
    test_size = int(0.1 * len(data))

    train = data.take(train_size)
    remaining = data.skip(train_size)
    val = remaining.take(val_size)
    test = remaining.skip(val_size)

    print(f"# train batchs = {len(train)}, # validate batchs = {len(val)}, # test batch = {len(test)}")


    if augmentation:
        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(factor=0.1),
            layers.RandomBrightness(factor=0.1),
            layers.GaussianNoise(stddev=0.1),
        ])
        train = train.map(lambda x, y: (data_augmentation(x), y))  # Apply augmentation only to the training set

    return train, val, test


def prepare_test_set(test_data):
    """
    Converts the test dataset into numpy arrays for easier evaluation.

    Args:
        test_data (tf.data.Dataset): The test dataset.

    Returns:
        dict: A dictionary containing the test images and labels as numpy arrays.
    """
    test_iter = test_data.as_numpy_iterator()
    test_set = {"images": tf.zeros((0, 224, 224, 3), dtype=tf.float32), "labels": tf.zeros((0,), dtype=tf.int32)}  # Use tf.zeros and specify dtype


    while True:
        try:
            batch = test_iter.next()
            test_set['images'] = tf.concat([test_set['images'], tf.convert_to_tensor(batch[0], dtype=tf.float32)], axis=0)
            test_set['labels'] = tf.concat([test_set['labels'], tf.convert_to_tensor(batch[1], dtype=tf.int32)], axis=0)  #Concatenate tensors
        except StopIteration:
            break

    return test_set



if __name__ == '__main__':
    # Example usage (requires data_loader.py)
    import data_loader  # Import data_loader

    data = data_loader.load_and_prepare_data('../input/eyediseasedata/dataset')
    train_data, val_data, test_data = preprocess_data(data)
    test_set = prepare_test_set(test_data)

    print("Data preprocessing complete.")
    print("Train dataset length:", len(train_data))
    print("Validation dataset length:", len(val_data))
    print("Test images shape:", test_set['images'].shape)
    print("Test labels shape:", test_set['labels'].shape)
