import os
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.preprocessing import image
from visualize_gradcam import make_gradcam_heatmap, save_and_display_gradcam


app = Flask(__name__)
CORS(app)  # ✅ Allow requests from React frontend

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ✅ Load the trained model
MODEL_PATH = "ai_detection_mobilenet.h5"
model = load_model(MODEL_PATH)
last_conv_layer_name = "Conv_1"  # Last conv layer in MobileNetV2

# ✅ Image processing function
IMG_SIZE = 224
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)  # Add batch dimension

# ✅ Route to handle image upload and prediction
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # ✅ Process image and predict
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)[0][0]

    # Adjust threshold for classification
    threshold = 0.6  # Adjust the threshold
    result = "Fake" if prediction < threshold else "Real"

    # ✅ Generate Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    cam_path = save_and_display_gradcam(file_path, heatmap, cam_path=os.path.join(STATIC_FOLDER, "cam.jpg"))

    return jsonify({
        "result": result,
        "heatmap_url": f"/static/{os.path.basename(cam_path)}"
    })

@app.route("/static/<filename>")
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)

# ✅ Run Flask server
if __name__ == "__main__":
    app.run(debug=True)
