Okay, here's a refactored version of your code, broken down into logical modules. I've focused on separating data loading, preprocessing, model definition, training, and evaluation into distinct files.  I've also tried to add comments to clarify the purpose of each module and function.

**Directory Structure:**

```
eye_disease_project/
├── data_loader.py
├── data_preprocessing.py
├── model_definition.py
├── model_training.py
├── model_evaluation.py
├── utils.py  #For helper functions
├── main.py       # Main execution script
└── README.md
```

**1. `data_loader.py`:**

```python
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
```

**2. `data_preprocessing.py`:**

```python
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
```

**3. `model_definition.py`:**

```python
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
```

**4. `model_training.py`:**

```python
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
```

**5. `model_evaluation.py`:**

```python
# model_evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_data, class_names):
    """
    Evaluates the trained model on the test data.

    Args:
        model (tf.keras.Model): The trained Keras model.
        test_data (dict): A dictionary containing the test images and labels as numpy arrays (from prepare_test_set).
        class_names (list): A list of class names.
    """

    y_true = test_data['labels']
    y_pred = np.argmax(model.predict(test_data['images']), axis=1)

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
    plt.xticks(np.arange(len(class_names)) + .5, class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)) + .5, class_names, rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_training_history(history):
    """Plots the training accuracy and loss curves."""
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=(15, 12))
    metrics = ['accuracy', 'loss']
    for i, metric in enumerate(metrics):
        plt.subplot(220 + 1 + i)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
    plt.show()


if __name__ == '__main__':
    # Example usage (requires data_loader.py, data_preprocessing.py, model_definition.py, model_training.py)
    import data_loader
    import data_preprocessing
    import model_definition
    import model_training

    data = data_loader.load_and_prepare_data('../input/eyediseasedata/dataset')
    train_data, val_data, test_data = data_preprocessing.preprocess_data(data)
    test_set = data_preprocessing.prepare_test_set(test_data)
    model = model_definition.create_model()
    history = model_training.train_model(model, train_data, val_data)

    class_names = data.class_names # Assuming data object has class_names
    evaluate_model(model, test_set, class_names)
    plot_training_history(history)
    print("Model evaluation complete.")
```

**6. `utils.py`:**

```python
# utils.py
import os
from subprocess import check_output
import pandas as pd
import numpy as np

def list_files(path):
    """Lists files in a given directory and prints the output."""
    try:
        output = check_output(["ls", path]).decode("utf8")
        print(output)
    except FileNotFoundError:
        print(f"Directory not found: {path}")

def get_class_names(data):
    """Extracts and prints the class names from a TensorFlow Dataset."""
    class_names = data.class_names
    for idx, name in enumerate(class_names):
        print(f"{idx} = {name}", end=", ")
    return class_names

def summarize_dataset(data):
    """Summarizes the value counts of labels in the dataset."""
    labels = np.concatenate([y for x, y in data], axis=0)
    values = pd.value_counts(labels)
    values = values.sort_index()
    print(values)

if __name__ == '__main__':
    # Example usage:
    DATASET_PATH = "../input/eyediseasedata/dataset"
    list_files(DATASET_PATH)

    # Create a dummy dataset for demonstration
    import tensorflow as tf
    dummy_data = tf.data.Dataset.from_tensor_slices((
        tf.random.uniform((10, 224, 224, 3), maxval=255, dtype=tf.int32),
        tf.random.uniform((10,), minval=0, maxval=4, dtype=tf.int32)
    ))
    dummy_data.class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']  # Add class_names attribute

    get_class_names(dummy_data)
    summarize_dataset(dummy_data)
```

**7. `main.py`:**

```python
# main.py
import data_loader
import data_preprocessing
import model_definition
import model_training
import model_evaluation
import utils  # Import the utils module

DATASET_PATH = "../input/eyediseasedata/dataset"  # Define dataset path

def main():
    """Main function to orchestrate the model training and evaluation."""

    # 1. Load the data
    print("Loading data...")
    data = data_loader.load_and_prepare_data(DATASET_PATH)
    utils.get_class_names(data)  # Print class names
    utils.summarize_dataset(data) # Summarize dataset labels

    # 2. Preprocess the data
    print("Preprocessing data...")
    train_data, val_data, test_data = data_preprocessing.preprocess_data(data, augmentation=True)
    test_set = data_preprocessing.prepare_test_set(test_data)


    # 3. Define the model
    print("Creating model...")
    model = model_definition.create_model()

    # 4. Train the model
    print("Training model...")
    history = model_training.train_model(model, train_data, val_data)

    # 5. Evaluate the model
    print("Evaluating model...")
    class_names = data.class_names
    model_evaluation.evaluate_model(model, test_set, class_names)
    model_evaluation.plot_training_history(history)


if __name__ == "__main__":
    main()
```

**Key Improvements and Explanations:**

*   **Modularity:** Each file now has a specific purpose, making the code easier to understand, maintain, and reuse.
*   **Functions:**  Code is encapsulated in functions, improving readability and testability.
*   **Clearer Arguments:**  Functions now accept arguments explicitly, making their purpose more obvious.  Configuration parameters (like `image_size`, `batch_size`, `epochs`, etc.) are now arguments, allowing you to easily change them.
*   **Docstrings:**  Each function has a docstring explaining what it does, what arguments it takes, and what it returns.
*   **`if __name__ == '__main__':` blocks:**  This ensures that the code in each file only runs when the file is executed directly, not when it's imported as a module.  This is good practice. I've included example usages in these blocks to show how to use the functions.
*   **Explicit Imports:** The main.py file imports all the modules
*   **Test Set Preparation**: Uses TensorFlow tensors to create test dataset efficiently and prevent errors during concatenation.
*   **Optimizer and Loss**: Specifies the learning rate for the Adamax optimizer and corrects the loss function to `sparse_categorical_crossentropy`

**How to Run:**

1.  **Create the directory structure:** Create the `eye_disease_project` directory and the files within it.
2.  **Place the dataset:** Make sure the `../input/eyediseasedata/dataset` path is correct, or adjust it accordingly.
3.  **Install Dependencies:** Make sure you have TensorFlow, scikit-learn, matplotlib, seaborn, and any other necessary libraries installed.  `pip install tensorflow scikit-learn matplotlib seaborn`
4.  **Run `main.py`:**  Execute the `main.py` script.  This will load the data, preprocess it, define the model, train it, and evaluate it.  `python main.py`

**Important Considerations:**

*   **Path Adjustments:**  You'll likely need to adjust the file paths (especially the path to the dataset) to match your specific environment.
*   **Error Handling:**  Add more robust error handling (e.g., `try...except` blocks) to handle potential issues like file not found errors or invalid data.
*   **Configuration:**  Consider using a configuration file (e.g., a JSON or YAML file) to store the various hyperparameters and file paths.  This makes it easier to change the configuration without modifying the code.
*   **Logging:** Add logging to track the progress of the training and evaluation process. This helps in debugging and monitoring.
*   **GPU Usage:**  If you have a GPU, make sure TensorFlow is configured to use it for faster training.
*   **Further Refinement:** This is a starting point.  You can further refine the code by adding more sophisticated data augmentation techniques, experimenting with different model architectures, and implementing more advanced evaluation metrics.
*   **Class Names**: Ensure that class names are available to the evaluation script.  The example now passes `data.class_names` (assuming this attribute exists). If it doesn't, you will need to get the class names another way (e.g., directly from the dataset directory).
*   **Data Types**: Make sure data types are consistent, especially when concatenating numpy arrays.

This refactored structure will make your project more organized, easier to maintain, and more scalable as you add more features. Remember to test each module individually as you develop.

