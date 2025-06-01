import os
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model  # type: ignore
import cv2
from werkzeug.utils import secure_filename
import traceback

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "efficientnet_finetuned_cifake.keras"
model = load_model(MODEL_PATH)

print("Model layers:")
for layer in model.layers:
    print(layer.name, layer.output.shape)

IMG_SIZE = 32
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

        return jsonify({
            "result": result,
            "confidence": float(prediction)
        })

    except Exception as e:
        print("Exception during /upload:")
        traceback.print_exc()  # print full error to console
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
