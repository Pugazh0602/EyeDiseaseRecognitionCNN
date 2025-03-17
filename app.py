# Importing necessary libraries
from flask import Flask, render_template, request  # Flask libraries for web framework
from tensorflow.keras.models import load_model  # For loading the pre-trained model
import numpy as np  # For numerical operations, especially with arrays
from tensorflow.keras.preprocessing.image import load_img  # To load the image for preprocessing
from tensorflow.keras.applications.vgg16 import preprocess_input  # To preprocess image for VGG16 model
import os  # For file path operations
from tensorflow.keras.preprocessing import image  # For image handling
from tensorflow.keras.utils import load_img, img_to_array  # For image preprocessing

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model (ensure the path is correct for your environment)
model = load_model('model/final_model.h5')

# Define the path to store uploaded images
target_img = os.path.join(os.getcwd(), 'static/images')

# Define allowed file extensions for uploaded images (png, jpg, jpeg)
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])

# Helper function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXT

# Function to load and preprocess the image for model input
def read_image(filename):
    # Load the image, resize it to 224x224 (required for VGG16 input size)
    img = load_img(filename, target_size=(224, 224))
    
    # Convert the image to a numpy array
    x = image.img_to_array(img)
    
    # Expand the dimensions to create a batch (needed for model input)
    x = np.expand_dims(x, axis=0)
    
    # Preprocess the image for the VGG16 model (scaling the pixel values)
    x = preprocess_input(x)
    
    # Return the processed image
    return x

# Define the route for the home page ('/')
@app.route('/', methods=['GET', 'POST'])
def predict():
    # Handle GET request (for displaying the form initially)
    if request.method == 'GET':
        return render_template('index.html', readImg='0')
    
    # Handle POST request (for receiving and processing the uploaded image)
    if request.method == 'POST':
        # Retrieve the file from the request
        file = request.files['filename']
        
        # Check if the file is valid (extension is allowed)
        if file and allowed_file(file.filename):
            filename = file.filename  # Get the filename
            file_path = os.path.join('static/images', filename)  # Set the path to save the file
            
            # Save the file to the server
            file.save(file_path)
            
            # Load the image for prediction
            image = load_img(file_path, target_size=(224, 224))  # Resize the image to 224x224
            input_arr = img_to_array(image)  # Convert the image to numpy array
            
            # Create a batch of size 1 (for prediction)
            input_arr = np.array([input_arr])
            
            # Run the image through the model to get the prediction
            prediction = model(input_arr)
            print(prediction)
            # Get the predicted class (the index of the class with the highest probability)
            classes_x = np.argmax(prediction, axis=1)
            
            # Flatten the prediction result (for easier display)
            pred = prediction.numpy().flatten()

            # Round the prediction to 2 decimal places
            np.set_printoptions(precision=4)
            pred = np.around(pred, 2)
            
            # Convert into percentage
            pred = pred*100
            
            # Define the list of possible labels for the classes
            labs = ['Cataract', 'Glaucoma', 'Dia_Ret', 'Normal']
            
            # Map the predicted class index to the corresponding disease label
            if classes_x == 0:
                diseases = "Cataract"
            elif classes_x == 2:
                diseases = "Glaucoma"
            elif classes_x == 1:
                diseases = "Diabetic_Retinopathy"
            else:
                diseases = "Normal"
            
            # Render the results page with the image, predicted disease, and prediction probabilities
            return render_template('index.html', 
                                   readImg='1', 
                                   diseases=diseases, 
                                   prob=pred, 
                                   user_image=file_path, 
                                   label=labs)
        else:
            # Return an error message if the file is not valid
            return "Unable to read the file. Please check file extension"

# Run the Flask app
if __name__ == '__main__':
    # Run the app with debug mode on and port set to 8000
    app.run(debug=True, use_reloader=True, port=8000)
