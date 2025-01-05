import os
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model
model = load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo[0][0]>0.5 :
        return "Brain tumor is present"
    else:
        return "Brain tumor is absent"

def getResult(img_path, model):
    # Load and preprocess the image
    image = cv2.imread(img_path)
    resized_image = cv2.resize(image, (64, 64))
    img = Image.fromarray(resized_image)
    img_array = np.array(img) / 255.0  # Normalize the pixel values
    input_img = np.expand_dims(img_array, axis=0)
    result = model.predict(input_img)

    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path, model)
        result = get_className(value)
        return result
    return "Prediction failed."

if __name__ == '__main__':
    app.run(debug=True)
