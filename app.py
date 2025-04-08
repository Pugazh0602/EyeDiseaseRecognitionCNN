import os
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import img_to_array, load_img

# Configuration
UPLOAD_FOLDER = 'static/uploaded-images'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MODEL_PATH = 'model/final_model.h5'
TARGET_SIZE = (224, 224)
LABELS = ['Cataract', 'Glaucoma', 'Dia_Ret', 'Normal']

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model (done only once at startup)
model = load_model(MODEL_PATH)

def allowed_file(filename):
    """Check if the filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size):
    """Load, resize, and preprocess the image for the model."""
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # VGG16 preprocessing
    return img_array

def predict_disease(model, img_array, labels):
    """Predict the disease from the image array and return the result."""
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = labels[predicted_class_index]
    probabilities = prediction.flatten()
    probabilities = np.around(probabilities * 100, 2)  # Convert to percentage and round
    return predicted_class, probabilities

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    """Handles image upload and prediction."""
    if request.method == 'GET':
        return render_template('index.html', readImg=False)

    if 'filename' not in request.files:
        return "No file part"

    file = request.files['filename']

    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img_array = preprocess_image(filepath, TARGET_SIZE)
        disease, probabilities = predict_disease(model, img_array, LABELS)

        return render_template(
            'index.html',
            readImg=True,
            diseases=disease,
            prob=probabilities,
            user_image=filepath,
            label=LABELS,
        )
    else:
        return "Invalid file type. Allowed types are: jpg, jpeg, png"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, port=8000)