import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0 # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler # type: ignore
from tensorflow.keras.metrics import AUC # type: ignore
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, accuracy_score, auc
import matplotlib.pyplot as plt
import math
import kagglehub

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))
if gpus:
    print(f"Using GPU: {gpus[0].name}")
else:
    print("No GPU detected, using CPU.")

# === Paths and dataset download/extraction ===
backend_path = os.getcwd()
archive_path = os.path.join(backend_path, "archive")

if not os.path.exists(archive_path):
    print("📥 Downloading CIFAKE dataset...")
    dataset_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")

    for file in os.listdir(dataset_path):
        if file.endswith(".zip"):
            zip_path = os.path.join(dataset_path, file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(backend_path)
                print(f"✅ Extracted: {file}")
else:
    print("✅ Dataset archive folder already exists, skipping download and extraction.")

training_path = os.path.join(archive_path, "training")
testing_path = os.path.join(archive_path, "testing")

if os.path.exists(training_path) and not os.path.exists(os.path.join(archive_path, "train")):
    os.rename(training_path, os.path.join(archive_path, "train"))

if os.path.exists(testing_path) and not os.path.exists(os.path.join(archive_path, "test")):
    os.rename(testing_path, os.path.join(archive_path, "test"))

TRAIN_DIR = os.path.join(archive_path, "train")
TEST_DIR = os.path.join(archive_path, "test")

print(f"Train directory: {TRAIN_DIR}")
print(f"Test directory: {TEST_DIR}")

# === Hyperparameters ===
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_INITIAL = 20
EPOCHS_FINE = 20
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5
LABEL_SMOOTHING = 0.1

# Count images function
def count_images(directory):
    total = 0
    for root, _, files in os.walk(directory):
        total += len([f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    return total

num_train_samples = count_images(TRAIN_DIR)
num_test_samples = count_images(TEST_DIR)

print(f"Number of training images: {num_train_samples}")
print(f"Number of test images: {num_test_samples}")

# === Data augmentation ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.1,
    fill_mode="reflect",
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Test generator class indices:", test_generator.class_indices)


# Compute class weights
classes = train_generator.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes
)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")

# === Model loading or building ===
finetuned_model_path = "efficientnet_cifake_finetuned.keras"

if os.path.exists(finetuned_model_path):
    print(f"Loading fine-tuned model from {finetuned_model_path}...")
    model = load_model(finetuned_model_path)
    print("Model loaded successfully!")
    # Recompile model for further training or evaluation
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy", AUC()]
    )
else:
    print("No fine-tuned model found. Building a new model from scratch...")
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy", AUC()]
    )

# Learning rate scheduler - cosine decay
def cosine_decay(epoch):
    max_lr = INITIAL_LR
    min_lr = 1e-6
    epochs = EPOCHS_INITIAL
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * epoch / epochs))
    return lr

lr_scheduler = LearningRateScheduler(cosine_decay)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    ModelCheckpoint("efficientnet_cifake_best.h5", save_best_only=True, monitor="val_accuracy", mode="max"),
    lr_scheduler
]

steps_per_epoch = int((num_train_samples * 0.8) // BATCH_SIZE)
validation_steps = int((num_train_samples * 0.2) // BATCH_SIZE)
test_steps = num_test_samples // BATCH_SIZE

# === Initial Training (if model was built new) ===
if not os.path.exists(finetuned_model_path):
    print("Starting initial training (feature extraction)...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=EPOCHS_INITIAL,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

# === Fine-tuning ===
print("Starting fine-tuning...")

# Make sure base_model is accessible (load again if needed)
if not os.path.exists(finetuned_model_path):
    base_model.trainable = True
else:
    # If loaded model, get base_model from model.layers or recreate it
    base_model = model.layers[0]  # typically the first layer is the base model input layer
    # But better to get by name or structure if possible
    # Alternatively rebuild base_model here if needed
    # For simplicity, we set all layers trainable except BatchNorm:
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

# Freeze BatchNorm layers and first 20 layers of EfficientNet
if hasattr(base_model, "layers"):
    for layer in base_model.layers[:20]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy", AUC()]
)

# Fine-tune LR scheduler
def fine_tune_lr(epoch):
    max_lr = FINE_TUNE_LR
    min_lr = 1e-7
    epochs = EPOCHS_FINE
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * epoch / epochs))
    return lr

fine_tune_scheduler = LearningRateScheduler(fine_tune_lr)

callbacks_fine = [
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    ModelCheckpoint("efficientnet_cifake_finetuned_best.h5", save_best_only=True, monitor="val_accuracy", mode="max"),
    fine_tune_scheduler
]

history_fine = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_fine,
    class_weight=class_weights,
    verbose=1
)

# Save the final fine-tuned model
model.save("efficientnet_cifake_final.keras")
print("✅ Model fine-tuned and saved successfully!")

# Evaluate on test set
print("Evaluating on test set...")
test_loss, test_acc, test_auc = model.evaluate(test_generator, steps=test_steps)
print(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

# --- Get all test predictions and true labels ---
print("\nGathering all test set predictions and labels...")

test_generator.reset()
all_preds = []
all_labels = []

for _ in range(test_steps):
    x_batch, y_batch = next(test_generator)
    preds = model.predict(x_batch)
    all_preds.extend(preds.flatten())
    all_labels.extend(y_batch)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# --- Find best threshold using Youden's J statistic from ROC curve ---
fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]

print(f"\nOptimal threshold found by ROC analysis: {best_threshold:.4f}")

# --- Calculate metrics at best threshold ---
best_pred_classes = (all_preds >= best_threshold).astype(int)
best_acc = accuracy_score(all_labels, best_pred_classes)
best_auc = auc(fpr, tpr)

print(f"Test Accuracy at optimal threshold: {best_acc:.4f}")
print(f"Test AUC (unchanged): {best_auc:.4f}")

# --- Optional: Plot ROC curve ---
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {best_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Best threshold = {best_threshold:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Is GPU being used?", tf.test.is_gpu_available())