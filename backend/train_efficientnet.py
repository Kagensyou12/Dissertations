import os
import zipfile
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.utils import class_weight
import kagglehub

# ✅ Download and extract dataset
print("Downloading CIFAKE dataset...")
dataset_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")

for file in os.listdir(dataset_path):
    if file.endswith(".zip"):
        zip_path = os.path.join(dataset_path, file)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
            print(f"✅ Extracted: {file}")

if os.path.exists(os.path.join(dataset_path, "training")):
    os.rename(os.path.join(dataset_path, "training"), os.path.join(dataset_path, "train"))
if os.path.exists(os.path.join(dataset_path, "testing")):
    os.rename(os.path.join(dataset_path, "testing"), os.path.join(dataset_path, "test"))

# ✅ Paths and Constants
TRAIN_DIR = os.path.join(dataset_path, "train")
TEST_DIR = os.path.join(dataset_path, "test")
IMG_SIZE = 224
BATCH_SIZE = 16

# ✅ Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    brightness_range=(0.8, 1.2),
    channel_shift_range=10.0,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ✅ Class Weights
classes = train_generator.classes
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(classes), y=classes)
class_weights = dict(enumerate(class_weights))

# ✅ Model Setup
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=x)

# ✅ Cosine Decay Learning Rate
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=3e-4,
    first_decay_steps=1000
)

# ✅ Compile Model
model.compile(optimizer=Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
              metrics=["accuracy"])

# ✅ Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("efficientnet_cifake_best.h5", save_best_only=True, monitor="val_accuracy", mode="max")
]

# ✅ Initial Training
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=5,
                    callbacks=callbacks,
                    class_weight=class_weights,
                    verbose=1)

# ✅ Fine-Tuning
base_model.trainable = True
for layer in base_model.layers[:50]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=3e-4),
              loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
              metrics=["accuracy"])

history_fine = model.fit(train_generator,
                         validation_data=val_generator,
                         epochs=5,
                         callbacks=callbacks,
                         class_weight=class_weights,
                         verbose=1)

# ✅ Save Model
model.save("efficientnet_cifake_finetuned.keras")
print("✅ EfficientNet model saved successfully!")

# ✅ Image Prediction
def predict_image(img_path):
    model = tf.keras.models.load_model("efficientnet_cifake_finetuned.keras")
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Real" if prediction < 0.5 else "Fake"
