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