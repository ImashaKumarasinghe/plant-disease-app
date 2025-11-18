import os
import json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Config
MODEL_PATH = 'plant_model.keras'
LABELS_PATH = 'labels.json'
INPUT_SIZE = (225, 225)   # must match training
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB upload limit

# Load model and labels once at startup
model = load_model(MODEL_PATH, compile=False)
with open(LABELS_PATH, 'r') as f:
    inv_labels = json.load(f)  # mapping like {"0":"Healthy", "1":"Powdery", "2":"Rust"}

def preprocess_image(image_path, target_size=INPUT_SIZE):
    img = load_img(image_path, target_size=target_size)
    arr = img_to_array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Save temporarily
    tmp_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(tmp_path)

    # Preprocess and predict
    x = preprocess_image(tmp_path)
    preds = model.predict(x)[0]
    class_idx = int(np.argmax(preds))
    class_name = inv_labels.get(str(class_idx), inv_labels.get(class_idx, 'Unknown'))

    # Clean up (optional)
    try:
        os.remove(tmp_path)
    except:
        pass

    return render_template('result.html', label=class_name, probs=preds.tolist())

if __name__ == '__main__':
    app.run(debug=True)
