# === Imports ===
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import os
import numpy as np
import zipfile
import torch
import kagglehub

# === Check GPU Availability ===
print("GPU Available:", torch.cuda.is_available())
print("TensorFlow GPU Device:", tf.test.gpu_device_name())

# === Set Seeds ===
tf.random.set_seed(42)
np.random.seed(42)

# === Dataset Setup and Paths ===
backend_path = os.getcwd()
archive_path = os.path.join(backend_path, "archive")

if not os.path.exists(archive_path):
    print("📥 Downloading CIFAKE dataset...")
    dataset_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
    for file in os.listdir(dataset_path):
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(dataset_path, file), 'r') as zip_ref:
                zip_ref.extractall(backend_path)
                print(f"✅ Extracted: {file}")
else:
    print("✅ Dataset archive exists, skipping download.")

def rename_if_exists(old, new):
    if os.path.exists(old) and not os.path.exists(new):
        os.rename(old, new)

rename_if_exists(os.path.join(archive_path, "training"), os.path.join(archive_path, "train"))
rename_if_exists(os.path.join(archive_path, "testing"), os.path.join(archive_path, "test"))

PATH = os.path.join(archive_path, "train")
TEST_PATH = os.path.join(archive_path, "test")
print(f"✅ Train dir: {PATH}, Test dir: {TEST_PATH}")

# === Constants ===
IMG_ROWS, IMG_COLS = 32, 32
LABELS = ['REAL', 'FAKE']
MODEL_PATH = "efficientnet_finetuned_cifake.keras"

# === Load Training Images ===
REAL_IMAGE, REAL_Y = [], []
realpath = os.path.join(PATH, 'REAL')
for path in os.listdir(realpath):
    image = tf.keras.preprocessing.image.load_img(os.path.join(realpath, path), target_size=(IMG_ROWS, IMG_COLS))
    image = tf.keras.preprocessing.image.img_to_array(image)
    REAL_IMAGE.append(image)
    REAL_Y.append(0)

FAKE_IMAGE, FAKE_Y = [], []
fakepath = os.path.join(PATH, 'FAKE')
for path in os.listdir(fakepath):
    image = tf.keras.preprocessing.image.load_img(os.path.join(fakepath, path), target_size=(IMG_ROWS, IMG_COLS))
    image = tf.keras.preprocessing.image.img_to_array(image)
    FAKE_IMAGE.append(image)
    FAKE_Y.append(1)

# === Load Test Images ===
TEST_REAL_IMAGE, TEST_REAL_Y = [], []
realpath = os.path.join(TEST_PATH, 'REAL')
for path in os.listdir(realpath):
    image = tf.keras.preprocessing.image.load_img(os.path.join(realpath, path), target_size=(IMG_ROWS, IMG_COLS))
    image = tf.keras.preprocessing.image.img_to_array(image)
    TEST_REAL_IMAGE.append(image)
    TEST_REAL_Y.append(0)

TEST_FAKE_IMAGE, TEST_FAKE_Y = [], []
fakepath = os.path.join(TEST_PATH, 'FAKE')
for path in os.listdir(fakepath):
    image = tf.keras.preprocessing.image.load_img(os.path.join(fakepath, path), target_size=(IMG_ROWS, IMG_COLS))
    image = tf.keras.preprocessing.image.img_to_array(image)
    TEST_FAKE_IMAGE.append(image)
    TEST_FAKE_Y.append(1)

# === Prepare Data ===
X = np.array(REAL_IMAGE + FAKE_IMAGE, dtype=np.float32) / 255.0
X = X.reshape((-1, IMG_ROWS, IMG_COLS, 3))

X_test = np.array(TEST_REAL_IMAGE + TEST_FAKE_IMAGE, dtype=np.float32) / 255.0
X_test = X_test.reshape((-1, IMG_ROWS, IMG_COLS, 3))

y_raw = np.array(REAL_Y + FAKE_Y)
Y_test = np.array(TEST_REAL_Y + TEST_FAKE_Y)

# === Train/Validation Split ===
X_train, X_val, Y_train, Y_val = train_test_split(X, y_raw, train_size=0.9)

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

# === Load or Train Model ===
if os.path.exists(MODEL_PATH):
    print(f"📦 Loading model from {MODEL_PATH}...")
    efficientnetB0 = load_model(MODEL_PATH)

    # === Continue Training for 10 Epochs ===
    efficientnetB0.compile(optimizer=keras.optimizers.Adam(1e-5),
                           loss=keras.losses.BinaryCrossentropy(),
                           metrics=[keras.metrics.BinaryAccuracy()])
    
    print("🔄 Continuing training for 10 more epochs...")
    efficientnetB0.fit(x=X_train, y=Y_train,
                       batch_size=32,
                       epochs=10,
                       validation_data=(X_val, Y_val))
else:
    print("🚀 Training model from scratch...")

    # Transfer learning base
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_ROWS, IMG_COLS, 3),
        pooling=None
    )
    base_model.trainable = False

    # Build model
    inputs = keras.Input(shape=(IMG_ROWS, IMG_COLS, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    efficientnetB0 = keras.Model(inputs, outputs)

    # Compile and train
    efficientnetB0.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.BinaryCrossentropy(),
                           metrics=[keras.metrics.BinaryAccuracy()])
    
    efficientnetB0.fit(x=X_train, y=Y_train,
                       batch_size=32,
                       epochs=20,
                       validation_data=(X_val, Y_val))

    # Fine-tune
    base_model.trainable = True
    efficientnetB0.compile(optimizer=keras.optimizers.Adam(1e-5),
                           loss=keras.losses.BinaryCrossentropy(),
                           metrics=[keras.metrics.BinaryAccuracy()])
    
    efficientnetB0.fit(x=X_train, y=Y_train,
                       batch_size=32,
                       epochs=10,
                       validation_data=(X_val, Y_val))
    
    # Save the trained model
    efficientnetB0.save(MODEL_PATH)
    print(f"💾 Model saved as {MODEL_PATH}")

# === Final Evaluation ===
loss, accuracy = efficientnetB0.evaluate(X_test, Y_test)
print(f"Final Model - Loss: {loss}, Accuracy: {accuracy}")

y_pred = efficientnetB0.predict(X_test)
print(classification_report(Y_test, y_pred.round()))
print("F1 Score:", f1_score(Y_test, y_pred.round()))
print("Recall:", recall_score(Y_test, y_pred.round()))
print("Precision:", precision_score(Y_test, y_pred.round()))

