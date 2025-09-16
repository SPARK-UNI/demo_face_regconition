from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from keras.models import load_model
import base64
import os

app = Flask(__name__)

# Load model and labels
print("Loading model...")
try:
    model = load_model("keras_model.h5")
    print("Model loaded successfully!")
    
    # Load labels from labels.txt
    with open("labels.txt", "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"Labels loaded: {labels}")
    
except Exception as e:
    print(f"Error loading model or labels: {e}")
    model = None
    labels = None

def predict_face_from_image(image_data):
    """
    Predict face from base64 image data
    """
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return "Invalid image"
        
        # Resize image to 224x224 (common for keras models from Teachable Machine)
        img = cv2.resize(frame, (224, 224))
        img = np.array(img, dtype=np.float32)
        
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        
        # Reshape for model input
        img = img.reshape(1, 224, 224, 3)
        
        # Predict
        pred = model.predict(img)
        class_idx = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred, axis=1)[0]
        
        # Get label name
        if class_idx < len(labels):
            result = labels[class_idx].split(' ', 1)[-1]  # Remove class number prefix if exists
            return f"{result} ({confidence:.2f})"
        else:
            return "Unknown"
        
    except Exception as e:
        return f"Error: {str(e)}"

# Route để serve index.html
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Route để serve CSS
@app.route('/style.css')
def css():
    return send_from_directory('.', 'style.css')

# Route để serve JavaScript
@app.route('/script.js')
def js():
    return send_from_directory('.', 'script.js')

# Route để serve static files (nếu có thêm ảnh, icon, v.v.)
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    # Handle actual prediction request
    try:
        if model is None or labels is None:
            response = jsonify({'prediction': 'Model not loaded', 'error': True})
        else:
            data = request.get_json()
            if not data or 'image' not in data:
                response = jsonify({'prediction': 'No image provided', 'error': True})
            else:
                image_data = data['image']
                prediction = predict_face_from_image(image_data)
                response = jsonify({'prediction': prediction, 'error': False})
        
        # Add CORS headers
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        response = jsonify({'prediction': f'Server error: {str(e)}', 'error': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

if __name__ == '__main__':
    print("Starting Face Recognition API...")
    print("Server running at: http://127.0.0.1:5000")
    print("Open browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)