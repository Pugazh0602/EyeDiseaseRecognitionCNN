# model_training.py
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adamax

def train_model(model, train_data, val_data, epochs=50, file_path="densenet_best_.hdf5"):
    """
    Trains the given Keras model.

    Args:
        model (tf.keras.Model): The Keras model to train.
        train_data (tf.data.Dataset): The training dataset.
        val_data (tf.data.Dataset): The validation dataset.
        epochs (int): The number of epochs to train for.
        file_path (str): The path to save the best model weights.

    Returns:
        tf.keras.callbacks.History: The training history.
    """
    model.compile(optimizer=Adamax(learning_rate=0.001),  # Explicit learning rate
                  loss='sparse_categorical_crossentropy',  # Corrected loss function
                  metrics=['accuracy'])


    checkpoint = ModelCheckpoint(file_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")  # save_best_only
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, mode="max") #restore_best_weights

    callbacks_list = [checkpoint, early_stopping]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks_list
    )

    return history


if __name__ == '__main__':
    # Example usage (requires data_loader.py, data_preprocessing.py, model_definition.py)
    import data_loader
    import data_preprocessing
    import model_definition

    data = data_loader.load_and_prepare_data('../input/eyediseasedata/dataset')
    train_data, val_data, test_data = data_preprocessing.preprocess_data(data)
    model = model_definition.create_model()

    history = train_model(model, train_data, val_data)

    print("Model training complete.")
