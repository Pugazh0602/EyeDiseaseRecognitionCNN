# data_loader.py
import tensorflow as tf

def load_and_prepare_data(directory, image_size=(224, 224), batch_size=64, seed=2022):
    """
    Loads data from a directory, prepares it for training, and returns a TensorFlow Dataset.

    Args:
        directory (str): Path to the directory containing the images.
        image_size (tuple):  The desired image size (height, width).
        batch_size (int): The batch size.
        seed (int):  Random seed for shuffling.

    Returns:
        tf.data.Dataset: A TensorFlow Dataset object.
    """

    data = tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed
    )

    return data

if __name__ == '__main__':
    #Example usage.  Make sure the path is correct.
    data = load_and_prepare_data('../input/eyediseasedata/dataset')
    print("Data loaded successfully.")
    for images, labels in data.take(1):
        print("Batch shape:", images.shape)
        print("Labels shape:", labels.shape)
