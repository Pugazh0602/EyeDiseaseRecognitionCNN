# Ocular Disease Recognition Web Application

This project implements a web application for automated ocular disease recognition using Convolutional Neural Networks (CNNs). The application allows users to upload an eye image, which is then processed by a pre-trained model to predict the presence of various ocular diseases, including Cataract, Glaucoma, Diabetic Retinopathy, and Normal.

## Project Structure

```
├── README.md               # This file
├── __pycache__            # Python cache directory (generated)
│   └── app.cpython-312.pyc # Compiled Python file
├── abstract.md           # Project abstract
├── app.py                  # Flask application code
├── final_model.h5          # Trained Keras model
├── fs                      # (Potentially empty directory - purpose unclear from this view)
├── model.txt               # (Purpose unclear from this view - potentially model architecture info or notes)
├── requirements.txt        # List of Python dependencies
├── static                 # Directory for static files (CSS, JavaScript, images)
│   ├── all.min.css        # Minified CSS file
│   ├── boxicons.min.css    # Minified CSS file
│   ├── eye.gif             # GIF image
│   ├── images             # Directory for images
│   │   ├── cat1.jpeg       # Example image
│   │   ├── dia.jpeg        # Example image
│   │   └── norm.jpg        # Example image
│   ├── script.js           # JavaScript file
│   ├── styles.css          # CSS file
│   ├── table.css           # CSS file
│   ├── table.html          # HTML file
│   └── table.js            # JavaScript file
├── templates              # Directory for HTML templates
│   ├── index.html          # Main HTML template
│   └── index2.html         # Secondary HTML template
└── testUi.py             # (Purpose unclear from this view - potentially UI testing script)
```

## Project Description

The `app.py` file contains the Flask application that serves as the backend for the web application.  It utilizes a pre-trained Keras model (`final_model.h5`) to predict ocular diseases. The `templates` directory contains the HTML files (`index.html`, `index2.html`) that define the user interface.  The `static` directory contains CSS, JavaScript, and image files to style and enhance the web application.

The application workflow is as follows:

1.  **User Upload:** The user uploads an eye image through the web interface.
2.  **Image Processing:** The uploaded image is saved to the `static/images` directory, preprocessed (resized to 224x224 and preprocessed for VGG16), and converted into a format suitable for the Keras model.
3.  **Prediction:** The preprocessed image is fed into the `final_model.h5` model for prediction.
4.  **Result Display:** The predicted disease and the probabilities associated with each class (Cataract, Glaucoma, Diabetic Retinopathy, Normal) are displayed to the user.

The `abstract.md` file contains a brief description of the project.  It highlights the importance of early eye disease detection and describes the CNN-based approach used in the project.

## Requirements

*   Python 3.6+
*   The libraries listed in `requirements.txt`.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate.bat # On Windows
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Note:  The `requirements.txt` specifies `tensorflow==1.12.0`.  This is a very old version of TensorFlow.  Consider upgrading to a more recent TensorFlow version (e.g., TensorFlow 2.x) if possible.  This might require adjustments to the model loading and prediction code in `app.py`.*

## Usage

1.  **Run the Flask application:**

    ```bash
    python app.py
    ```

2.  **Open the application in your web browser:**

    Go to `http://127.0.0.1:8000/` or `http://localhost:8000/`.

3.  **Upload an eye image:**

    Use the file upload form to select an image of an eye.

4.  **View the results:**

    The application will display the predicted disease and the probabilities for each disease class.

## Model

The pre-trained model (`final_model.h5`) is a crucial component of this application.  The model architecture and training details are not fully clear from the provided file structure, but based on the `app.py` code, it seems to be based on the VGG16 architecture.

## Considerations

*   **TensorFlow Version:**  The application currently uses TensorFlow 1.12.0, which is outdated.  Consider upgrading to a more recent version of TensorFlow for improved performance and compatibility. This will require changes in the model loading/saving and image processing code.
*   **Model Accuracy:**  The accuracy of the ocular disease recognition depends on the quality and size of the training dataset used to train the `final_model.h5` model.
*   **User Interface:** The user interface (HTML templates and CSS) can be further improved to provide a better user experience.  The Javascript (`script.js`, `table.js`) files can add interactivity to the page.
*   **Error Handling:**  The application includes basic error handling for file uploads, but more robust error handling can be added to improve the application's reliability.
*   **Security:** Implement security best practices for web applications, such as input validation and protection against cross-site scripting (XSS) and SQL injection vulnerabilities.

## Contributing

Contributions to this project are welcome.  Please fork the repository and submit a pull request with your changes.

## License

[Specify the license under which this project is released. For example, MIT License.]
