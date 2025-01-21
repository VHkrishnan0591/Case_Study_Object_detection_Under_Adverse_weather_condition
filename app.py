from flask import Flask, jsonify, request, render_template
import os
from PIL import Image, ImageDraw
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from Image_preprocessing import ImageDehazer, ImageDerained
import matplotlib.pyplot as plt
import os
from yolopredict import yolo_predict

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return jsonify({'uploaded_image': f'/{file_path}'})

@app.route('/predict', methods=['POST'])
def predict():
    # Use the last uploaded image
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
    if not uploaded_files:
        return jsonify({'error': 'No image uploaded yet'}), 400

    latest_image = max(
        [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in uploaded_files],
        key=os.path.getctime
    )

    # Load the model
    model = load_model('fog_rain_normal.h5')
    yolopredict = yolo_predict()
    # Read the input image
    image = cv2.imread(latest_image)
    image1 = cv2.resize(image, (150,150))
    image2 = image1.astype('float32') / 255.0 
    image_array = image2.reshape(1, 150, 150, 3) 
    # Perform prediction using the loaded model and the input image
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)  # Get the class with the highest probability
    print(f'{latest_image} Predicted class: {predicted_class}')
    if predicted_class[0] == 1:
        img_derained = ImageDerained()
        final_image = img_derained.derainer(image)
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        results = yolopredict.predict_and_compare(final_image,image)
    elif predicted_class[0] == 0:
        Imagedehazer = ImageDehazer()
        final_image = Imagedehazer.dehazer(image)
        results = yolopredict.predict_and_compare(final_image,image)
    else:
        results = yolopredict.predict_and_compare(image,image)
    
    # Save the result image
    result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result_YOLO_Custom.jpg')
    cv2.imwrite(result_image_path, results[0])
    result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result_YOLO_Pretrained.jpg')
    cv2.imwrite(result_image_path, results[1])


    return jsonify({
        'prediction_images': [
            '/static/results/result_YOLO_Pretrained.jpg',
            '/static/results/result_YOLO_Custom.jpg'
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)
