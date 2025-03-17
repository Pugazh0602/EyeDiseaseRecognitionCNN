```python
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Conv2D,
    GlobalAveragePooling2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Dense,
    BatchNormalization,
    Input,
)
from keras.models import Model
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121


DATASET_PATH = "../input/eyediseasedata/dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
SEED = 2022
NUM_CLASSES = 4
EARLY_STOPPING_PATIENCE = 10
NUM_EPOCHS = 50
DENSENET_FROZEN_LAYERS = 121


def load_and_preprocess_data(dataset_path, image_size, batch_size, seed):
    data = tf.keras.utils.image_dataset_from_directory(
        directory=dataset_path,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
    )
    data = data.map(lambda x, y: (x / 255, y))
    return data


def split_data(data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    train_size = int(train_ratio * len(data)) + 1
    val_size = int(val_ratio * len(data))
    test_size = int(test_ratio * len(data))

    train = data.take(train_size)
    remaining = data.skip(train_size)
    val = remaining.take(val_size)
    test = remaining.skip(val_size)

    return train, val, test


def create_data_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(factor=0.1),
            layers.RandomBrightness(factor=0.1),
            layers.GaussianNoise(stddev=0.1),
        ]
    )


def prepare_test_set(test_data):
    test_iter = test_data.as_numpy_iterator()
    test_set = {"images": np.empty((0, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)), "labels": np.empty(0)}
    while True:
        try:
            batch = test_iter.next()
            test_set["images"] = np.concatenate((test_set["images"], batch[0]))
            test_set["labels"] = np.concatenate((test_set["labels"], batch[1]))
        except StopIteration:
            break
    return test_set


def build_model(input_shape, num_classes, frozen_layers):
    dense = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in dense.layers[:frozen_layers]:
        layer.trainable = False

    x = dense.output
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    pred = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=dense.input, outputs=pred)
    return model


def compile_model(model):
    model.compile(
        optimizer="adamax", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )


def train_model(model, train_data, val_data, file_path, patience, epochs):
    checkpoint = ModelCheckpoint(
        file_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
    )
    early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    callbacks_list = [checkpoint, early_stopping]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks_list,
    )
    return history


def evaluate_model(model, test_data):
    return model.evaluate(test_data)


def plot_training_history(history):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(15, 12))
    metrics = ["accuracy", "loss"]
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


def generate_classification_report(model, test_set, class_names):
    y_true = test_set["labels"]
    y_pred = np.argmax(model.predict(test_set["images"]), 1)
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_confusion_matrix(model, test_set, class_names):
    y_true = test_set["labels"]
    y_pred = np.argmax(model.predict(test_set["images"]), 1)
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
    data = load_and_preprocess_data(DATASET_PATH, IMAGE_SIZE, BATCH_SIZE, SEED)
    class_names = data.class_names
    train_data, val_data, test_data = split_data(data)
    data_augmentation = create_data_augmentation()
    train_data = train_data.map(lambda x, y: (data_augmentation(x), y))
    test_set = prepare_test_set(test_data)
    model = build_model(IMAGE_SIZE + (3,), NUM_CLASSES, DENSENET_FROZEN_LAYERS)
    compile_model(model)

    file_path = "densenet_best_.hdf5"
    history = train_model(
        model, train_data, val_data, file_path, EARLY_STOPPING_PATIENCE, NUM_EPOCHS
    )

    evaluate_model(model, test_data)
    plot_training_history(history)
    generate_classification_report(model, test_set, class_names)
    plot_confusion_matrix(model, test_set, class_names)


if __name__ == "__main__":
    main()
```
