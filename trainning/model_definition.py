# model_definition.py
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization

def create_model(num_classes=4, trainable_layers=121, image_size=(224, 224, 3)):
    """
    Creates a DenseNet121-based model for image classification.

    Args:
        num_classes (int): The number of classes to predict.
        trainable_layers (int):  Number of layers to freeze in the base DenseNet121 model.
        image_size (tuple): The size of the input images (height, width, channels).

    Returns:
        tf.keras.Model: The compiled Keras model.
    """

    dense = DenseNet121(weights="imagenet", include_top=False, input_shape=image_size)

    # Freeze layers
    for layer in dense.layers[:trainable_layers]:
        layer.trainable = False

    model = Sequential()
    model.add(dense)
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))  # Changed to softmax for multi-class

    return model

if __name__ == '__main__':
    # Example usage
    model = create_model()
    print("Model created successfully.")
    model.summary()
