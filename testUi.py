from flask import Flask, render_template, request
import os

app = Flask(__name__)
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])
INDEX_FILE="index.html"
IMG_SAVE_PATH='static/uploaded-images'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXT

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template(INDEX_FILE, readImg=False)
    
    if request.method == 'POST':
        file = request.files['filename']
        if file and allowed_file(file.filename):
            filename = file.filename  # Get the filename
            file_path = os.path.join(IMG_SAVE_PATH, filename)  # Set the path to save the file
            file.save(file_path)
            pred = [.7,.4,.1,.02]
            labs = ['Cataract', 'Glaucoma', 'Diabetic Retinopathy', 'Normal']
            return render_template(INDEX_FILE, 
                                   readImg=True, 
                                   diseases="Diabetic Retinopathy", 
                                   prob=pred, 
                                   user_image=file_path, 
                                   label=labs)
        else:
            return "Unable to read the file. Please check file extension"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, port=8000)