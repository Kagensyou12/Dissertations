import os
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model # type: ignore
import cv2
from visualize_gradcam import make_gradcam_heatmap, save_and_display_gradcam
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

MODEL_PATH = "efficientnet_cifake_finetuned.keras"
model = load_model(MODEL_PATH)

last_conv_layer_name = "top_conv"
IMG_SIZE = 224
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)[0][0]

        threshold = 0.5
        result = "Real" if prediction < threshold else "Fake"
        print(f"Prediction score: {prediction}")

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        cam_path = os.path.join(STATIC_FOLDER, f"cam_{filename}")
        save_and_display_gradcam(file_path, heatmap, cam_path=cam_path)

        return jsonify({
            "result": result,
            "heatmap_url": f"/static/{os.path.basename(cam_path)}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/static/<filename>")
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
