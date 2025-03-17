```python
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from keras.models import Model
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121


def build_dataset(directory, batch_size, image_size, seed):
    return tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
    )


def preprocess_data(data):
    return data.map(lambda x, y: (x / 255, y))


def split_dataset(data, train_size, val_size, test_size):
    train = data.take(train_size)
    remaining = data.skip(train_size)
    val = remaining.take(val_size)
    test = remaining.skip(val_size)
    return train, val, test


def augment_data():
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


def prepare_test_set(test):
    test_iter = test.as_numpy_iterator()
    test_set = {"images": np.empty((0, 224, 224, 3)), "labels": np.empty(0)}
    while True:
        try:
            batch = test_iter.next()
            test_set["images"] = np.concatenate((test_set["images"], batch[0]))
            test_set["labels"] = np.concatenate((test_set["labels"], batch[1]))
        except StopIteration:
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    return test_set


def build_model(input_shape=(224, 224, 3), num_classes=4):
    dense = DenseNet121(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    for layer in dense.layers[:121]:
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


def compile_model(model, optimizer="adamax", loss="sparse_categorical_crossentropy", metrics=["accuracy"]):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def train_model(model, train, val, epochs=50, filepath="densenet_best_.hdf5", patience=10):
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_accuracy", verbose=1, save_best_only=True,
        save_weights_only=False, mode="max"
    )
    early = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    callbacks_list = [checkpoint, early]
    history = model.fit(
        train, validation_data=val, epochs=epochs, callbacks=callbacks_list
    )
    return history


def plot_history(history):
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


def evaluate_model(model, test_set, class_names):
    y_true = test_set["labels"]
    y_pred = np.argmax(model.predict(test_set["images"]), 1)
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, vmin=0, fmt="g", cmap="Blues", cbar=False)
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    DATASET_DIR = "../input/eyediseasedata/dataset"
    BATCH_SIZE = 64
    IMAGE_SIZE = (224, 224)
    SEED = 2022
    EPOCHS = 50
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.2
    TEST_SIZE = 0.1

    print(f"Dataset directory content: {os.listdir(DATASET_DIR)}")
    data = build_dataset(DATASET_DIR, BATCH_SIZE, IMAGE_SIZE, SEED)

    class_names = data.class_names
    print("Class Names:", ", ".join([f"{idx} = {name}" for idx, name in enumerate(class_names)]))

    data = preprocess_data(data)
    train_size = int(TRAIN_SIZE * len(data)) + 1
    val_size = int(VAL_SIZE * len(data))
    test_size = int(TEST_SIZE * len(data))

    train, val, test = split_dataset(data, train_size, val_size, test_size)

    print(f"# train batchs = {len(train)}, # validate batchs = {len(val)}, # test batch = {len(test)}")

    data_augmentation = augment_data()
    train = train.map(lambda x, y: (data_augmentation(x), y))
    test_set = prepare_test_set(test)

    model = build_model()
    compile_model(model)
    history = train_model(model, train, val, epochs=EPOCHS)
    model.evaluate(test)
    plot_history(history)
    evaluate_model(model, test_set, class_names)
```
