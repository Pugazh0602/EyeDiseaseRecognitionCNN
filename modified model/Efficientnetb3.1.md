```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from subprocess import check_output
import warnings

warnings.filterwarnings("ignore")


def load_data(directory, image_size=(224, 224), batch_size=64, seed=2022):
    """Loads image data from a directory using tf.keras.utils.image_dataset_from_directory."""
    return tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
    )


def preprocess_data(dataset):
    """Normalizes image data to the range [0, 1]."""
    return dataset.map(lambda x, y: (x / 255, y))


def split_data(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Splits a dataset into training, validation, and test sets."""
    train_size = int(train_ratio * len(dataset)) + 1
    val_size = int(val_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))

    train = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val = remaining.take(val_size)
    test = remaining.skip(val_size)

    return train, val, test


def prepare_test_set(test_data):
    """Converts a test dataset into numpy arrays for images and labels."""
    test_iter = test_data.as_numpy_iterator()
    test_set = {"images": np.empty((0, 224, 224, 3)), "labels": np.empty(0)}
    while True:
        try:
            batch = test_iter.next()
            test_set["images"] = np.concatenate((test_set["images"], batch[0]))
            test_set["labels"] = np.concatenate((test_set["labels"], batch[1]))
        except StopIteration:
            break
    return test_set


def create_efficientnet_model(num_classes):
    """Creates an EfficientNetB3 model with a custom classification head."""
    effnet = EfficientNetB3(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="max"
    )
    effnet.trainable = False

    x = effnet.output
    x = BatchNormalization()(x)
    x = Dense(
        1024,
        kernel_regularizer=regularizers.l2(l=0.016),
        activity_regularizer=regularizers.l1(0.006),
        bias_regularizer=regularizers.l1(0.006),
        activation="relu",
    )(x)
    x = Dropout(rate=0.45, seed=2022)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=effnet.input, outputs=output)
    model.compile(
        optimizer="adamax", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_model(model, train_data, val_data, epochs=50, filepath="efficient_best_.hdf5"):
    """Trains the model with early stopping and model checkpointing."""
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
    )
    early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", patience=10)
    callbacks_list = [checkpoint, early_stopping]

    history = model.fit(
        train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks_list
    )
    return history


def plot_training_history(history):
    """Plots the training history (accuracy and loss)."""
    plt.figure(figsize=(15, 12))
    metrics = ["accuracy", "loss"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, metric in enumerate(metrics):
        plt.subplot(220 + 1 + i)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label="Train")
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            color=colors[1],
            linestyle="--",
            label="Val",
        )
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
    plt.show()


def evaluate_model(model, test_data):
    """Evaluates the model on the test data."""
    return model.evaluate(test_data)


def generate_predictions(model, test_images):
    """Generates predictions from the model on the test images."""
    return np.argmax(model.predict(test_images), axis=1)


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, vmin=0, fmt="g", cmap="Blues", cbar=False)
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def main():
    """Main function to execute the image classification pipeline."""
    data_dir = "../input/eyediseasedata/dataset"
    print(check_output(["ls", data_dir]).decode("utf8"))
    print(os.listdir(data_dir))

    data = load_data(data_dir)
    class_names = data.class_names
    num_classes = len(class_names)
    print(f"Class names: {class_names}")

    data = preprocess_data(data)
    train_data, val_data, test_data = split_data(data)

    test_set = prepare_test_set(test_data)
    y_true = test_set["labels"]

    model = create_efficientnet_model(num_classes)
    history = train_model(model, train_data, val_data)
    plot_training_history(history)

    evaluation = evaluate_model(model, test_data)
    print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

    y_pred = generate_predictions(model, test_set["images"])
    print(classification_report(y_true, y_pred, target_names=class_names))
    plot_confusion_matrix(y_true, y_pred, class_names)

    model.save_weights("effmodel_weights.h5")
    model.save("effmodel_keras.h5")


if __name__ == "__main__":
    main()
```
