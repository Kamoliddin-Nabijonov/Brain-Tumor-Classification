import requests
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask('Brain-Tumor-Classification')
model = load_model('./Trained_Models/Sequential_13_0.999.keras')


def prepare_image_from_url(img_url):
    try:
        response = requests.get(img_url)
        
        if 'image' not in response.headers.get('Content-Type', ''):
            return None, 'The provided URL does not point to a valid image.'
        
        img = Image.open(BytesIO(response.content))  
        img = img.convert("RGB")
        img = img.resize((150, 150))  
        img = np.array(img) / 255.0 
        img = np.expand_dims(img, axis=0)
        return img, None
    except Exception as e:
        return None, f"Error processing image: {str(e)}"


@app.route('/predict', methods=['POST'])
def predict():
    if 'image_url' not in request.json:
        return jsonify({'error': 'No URL to image provided'}), 400
    
    img_url = request.json['image_url']
    
    if not img_url:
        return jsonify({'error': 'Invalid URL'}), 400

    img, error = prepare_image_from_url(img_url)
    
    if img is None:
        return jsonify({'error': error}), 500

    try:
        prediction = model.predict(img)
        result = 'Positive' if prediction[0] > 0.5 else 'Negative'  # Binary classification
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
