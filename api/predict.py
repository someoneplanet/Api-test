# api/predict.py

import requests
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
import pytesseract
from http import HTTPStatus

# Load pre-trained ResNet model for image classification
model = ResNet50(weights='imagenet')

# Function to get predictions
def get_image_prediction(image_url):
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))

    # Resize and preprocess the image
    img = img.resize((224, 224))  # ResNet50 takes 224x224 images
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
    img_array = preprocess_input(img_array)

    # Predict the class
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    # Return top prediction label
    return decoded_predictions[0][1]

# Function to extract text using Tesseract OCR
def extract_text(image_url):
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))
    text = pytesseract.image_to_string(img)
    return text.strip()

# Serverless function to handle the prediction request
def handler(request):
    # Get the image URL from query parameters
    image_url = request.args.get('url')
    if not image_url:
        return {"error": "No URL provided"}, HTTPStatus.BAD_REQUEST

    # Try extracting text first (OCR)
    text = extract_text(image_url)
    if text:
        return {"text": text}, HTTPStatus.OK

    # If no text, use the image classification model
    prediction = get_image_prediction(image_url)
    return {"prediction": prediction}, HTTPStatus.OK
