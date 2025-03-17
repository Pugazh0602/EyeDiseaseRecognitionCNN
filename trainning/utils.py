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
