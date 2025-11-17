#!/usr/bin/env python3
r"""
General training script that detects all class folders under
`data/dataset/Train/Train` and trains a MobileNetV2 classifier.
Saves model to `models/plant_disease_model.h5` and updates metadata.

Run inside the project's `.venv`:
Use `.venv/Scripts/Activate.ps1` (or escape backslashes if using the path in Python strings)
Example:
    .venv\Scripts\Activate.ps1
    python train_all.py
"""

import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Configuration
DATA_DIR = "data/dataset/Train/Train"
MODEL_PATH = "models/plant_disease_model.h5"
METADATA_PATH = "models/training_results_results.json"
CLASSES_PATH = "classes.json"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# Auto-detect class folders
classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
num_classes = len(classes)

print(f"Detected {num_classes} classes: {classes}")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Build model
base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# Train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    verbose=1
)

# Save model
model.save(MODEL_PATH)
print(f"Saved model to: {MODEL_PATH}")

# Save metadata
metadata = {
    "class_names": classes,
    "num_classes": num_classes,
    "training_accuracy": float(history.history['accuracy'][-1]),
    "image_size": IMAGE_SIZE,
    "model_type": "MobileNetV2",
    "epochs_trained": EPOCHS
}

with open(METADATA_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata to: {METADATA_PATH}")

# Update classes.json
classes_data = {
    "class_names": classes,
    "num_classes": num_classes,
    "accuracy": float(history.history['accuracy'][-1])
}

with open(CLASSES_PATH, 'w') as f:
    json.dump(classes_data, f, indent=2)
print(f"Updated {CLASSES_PATH}")

print("Training finished")
