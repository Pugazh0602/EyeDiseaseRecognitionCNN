# Ocular Disease Recognition Web Application

This project implements a web application for automated ocular disease recognition using Convolutional Neural Networks (CNNs) a Deep Learning Tech. The application allows users to upload an eye image, which is then processed by a pre-trained model to predict the presence of various ocular diseases, including Cataract, Glaucoma, Diabetic Retinopathy, and Normal.p

## Project Description

The `app.py` file contains the Flask application that serves as the backend for the web application.  It utilizes a pre-trained Keras model (`final_model.h5`) to predict ocular diseases. The `templates` directory contains the HTML files (`index.html`, `index2.html`) that define the user interface.  The `static` directory contains CSS, JavaScript, and image files to style and enhance the web application.

The application workflow is as follows:

1.  **User Upload:** The user uploads an eye image through the web interface.
2.  **Image Processing:** The uploaded image is saved to the `static/images` directory, preprocessed (resized and potentially preprocessed for the specific model - see model details below), and converted into a format suitable for the Keras model.
3.  **Prediction:** The preprocessed image is fed into the `final_model.h5` model for prediction.
4.  **Result Display:** The predicted disease and the probabilities associated with each class (Cataract, Glaucoma, Diabetic Retinopathy, Normal) are displayed to the user.

The `abstract.md` file contains a brief description of the project.  It highlights the importance of early eye disease detection and describes the CNN-based approach used in the project.

## Model Details

The `final_model.h5` file contains the pre-trained model. The architecture is summarized in `model.txt` and consists of the following layers:

*   **Base Model:** DenseNet121 (pre-trained, likely on ImageNet, and then fine-tuned for this task).  The output is then passed through a Max Pooling layer.
*   **Batch Normalization:**  Normalizes the output of the pooling layer for better training stability.
*   **Dropout:**  Regularization technique to prevent overfitting.
*   **Flatten:**  Converts the 3D feature maps into a 1D vector.
*   **Dense Layers:** Two fully connected (Dense) layers with 512 neurons and ReLU activation, followed by a Dropout layer.
*   **Output Layer:**  A final Dense layer with 4 neurons (corresponding to the four classes: Cataract, Glaucoma, Diabetic Retinopathy, Normal) and a Softmax activation function (implicit in the `app.py` code).

## Requirements

*   Python 3.6+
*   The libraries listed in `requirements.txt`.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Pugazh0602/EyeDiseaseRecognitionCNN.git
    cd EyeDiseaseRecognitionCNN
    ```

2.  **Create a virtual environment (recommended):**

    ```
    python -m venv venv
    ```
    ```bash
    source venv/bin/activate  # On Linux/macO
    ```
    or
    ```bat
    venv\Scripts\activate.bat # On Windows
    ```

3.  **Install the dependencies:**

    ```
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Flask application:**

    ```
    python app.py
    #or
    python testUi.py #for testing ui
    ```

2.  **Open the application in your web browser:**

    Go to `http://127.0.0.1:8000/` or `http://localhost:8000/`.

3.  **Upload an eye image:**

    Use the file upload form to select an image of an eye.

4.  **View the results:**

    The application will display the predicted disease and the probabilities for each disease class.

## Thanks
Thanks to [sanjays50](https://github.com/sanjays50)'s project [Eye-Disease-Classigication-and-Detection](https://github.com/sanjays50/Eye-Disease-Classigication-and-Detection) for a head start.

## License
This project uses [LICENSE](LICENSE)
