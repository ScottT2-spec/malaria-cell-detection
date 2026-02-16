"""
Malaria Cell Detection — Web Application
Scan blood cell images via camera for AI-powered malaria screening
Uses ONNX Runtime for lightweight inference
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import base64
import os

app = Flask(__name__)

# ─── Load Model ───
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'malaria_model.onnx')
session = None

def load_model():
    global session
    if os.path.exists(MODEL_PATH):
        session = ort.InferenceSession(MODEL_PATH)
        print(f"ONNX model loaded: {MODEL_PATH}")
        inp = session.get_inputs()[0]
        print(f"Input: {inp.name}, shape: {inp.shape}, type: {inp.type}")
    else:
        print(f"Model not found at {MODEL_PATH}")
        print("Export model from Kaggle notebook first!")

IMG_SIZE = 128

def preprocess_image(image_data):
    """Decode base64 image, resize, normalize"""
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 128, 128, 3)
    return arr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image received'}), 400
    
    try:
        img = preprocess_image(data['image'])
        
        inp_name = session.get_inputs()[0].name
        result = session.run(None, {inp_name: img})
        
        prediction = float(result[0][0])
        
        # Handle both single-output sigmoid and two-output softmax
        if hasattr(result[0][0], '__len__') and len(result[0][0]) == 2:
            parasitized_prob = float(result[0][0][0])
            uninfected_prob = float(result[0][0][1])
        else:
            # Single sigmoid: 0 = Parasitized, 1 = Uninfected
            uninfected_prob = float(result[0][0])
            if uninfected_prob > 1:  # logit, apply sigmoid
                uninfected_prob = 1 / (1 + np.exp(-uninfected_prob))
            parasitized_prob = 1 - uninfected_prob
        
        is_malaria = parasitized_prob > 0.5
        
        return jsonify({
            'parasitized': round(parasitized_prob * 100, 1),
            'uninfected': round(uninfected_prob * 100, 1),
            'diagnosis': 'Parasitized — Malaria Detected ⚠️' if is_malaria else 'Uninfected — No Malaria ✅',
            'confidence': round(max(parasitized_prob, uninfected_prob) * 100, 1),
            'is_malaria': is_malaria
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'running', 'model': session is not None})

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5001, debug=False)
