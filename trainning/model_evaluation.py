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
