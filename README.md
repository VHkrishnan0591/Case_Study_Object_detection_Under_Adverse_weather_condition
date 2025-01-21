# README: Flask Application for Image Processing and Weather Classification

## Overview
This Flask-based web application performs image processing and weather classification. Users can upload images of outdoor scenes, and the application predicts weather conditions (foggy, rainy, or normal). Based on the prediction, it applies appropriate pre-processing (dehazing or deraining). The processed image is then analyzed using a YOLO-based object detection model, and the results are displayed.

## Features
- Upload images through a web interface.
- Predict weather conditions using a deep learning model.
- Apply deraining or dehazing based on predictions.
- Perform object detection using both a custom YOLO model and a pretrained YOLO model.
- Display processed images and prediction results.

---

## Installation and Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.8+

### Steps
1. Unzip this zip file

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. use the training files for retraing the models.

4. Start the Flask application:
    ```bash
    python app.py
    ```

5. Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## How It Works
1. **Image Upload**: Users upload an image through the web interface.
2. **Weather Classification**: The `fog_rain_normal.h5` model predicts whether the weather is foggy, rainy, or normal.
3. **Image Preprocessing**:
   - If foggy, the `ImageDehazer` class processes the image to remove haze.
   - If rainy, the `ImageDerained` class processes the image to remove rain.
   - If normal, the image is left unchanged.
4. **Object Detection**: The processed image is passed to two YOLO models (custom and pretrained).
5. **Results**: The detected objects are marked on the images, and results are saved in `static/results`.

---
