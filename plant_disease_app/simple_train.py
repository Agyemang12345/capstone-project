#!/usr/bin/env python3
"""
Simple Training Script - Plant Disease Detection
Run this to train your model
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("🌿 PLANT DISEASE DETECTION - SIMPLE TRAINING SCRIPT")
print("=" * 80)

# Configuration
DATA_DIR = "data/dataset/Train/Train"
MODEL_NAME = "plant_disease_model.h5"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

print(f"\n📁 Data directory: {DATA_DIR}")
print(f"🖼️  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"📊 Batch size: {BATCH_SIZE}")
print(f"🔄 Epochs: {EPOCHS}")

# Step 1: Load data
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

images = []
labels = []
class_names = []

# Get class directories
class_dirs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"\n✓ Found classes: {class_names}")

for class_idx, class_name in enumerate(class_dirs):
    class_dir = os.path.join(DATA_DIR, class_name)
    class_names.append(class_name)
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"  Loading {class_name}... ", end="")
    for img_file in image_files:
        img_path = os.path.join(class_dir, img_file)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # Normalize
                images.append(img)
                labels.append(class_idx)
        except:
            pass
    print(f"✓ {len([l for l in labels if l == class_idx])} images")

print(f"\n✓ Total images loaded: {len(images)}")
print(f"✓ Classes: {class_names}")

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Step 2: Split data
print("\n" + "="*80)
print("STEP 2: SPLITTING DATA")
print("="*80)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\n✓ Training set: {X_train.shape[0]} images")
print(f"✓ Validation set: {X_val.shape[0]} images")
print(f"✓ Test set: {X_test.shape[0]} images")

# Step 3: Build model
print("\n" + "="*80)
print("STEP 3: BUILDING MODEL")
print("="*80)

print("\n📦 Creating MobileNetV2 model...")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Add custom layers
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

print(f"✓ Model created")
print(f"✓ Output classes: {len(class_names)}")

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled")

# Step 4: Train model
print("\n" + "="*80)
print("STEP 4: TRAINING MODEL")
print("="*80)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\n✓ Training complete!")

# Step 5: Evaluate model
print("\n" + "="*80)
print("STEP 5: EVALUATING MODEL")
print("="*80)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✓ Test Accuracy: {test_acc*100:.2f}%")
print(f"✓ Test Loss: {test_loss:.4f}")

# Step 6: Save model
print("\n" + "="*80)
print("STEP 6: SAVING MODEL")
print("="*80)

model.save(MODEL_NAME)
print(f"\n✓ Model saved: {MODEL_NAME}")

# Step 7: Save class names
import json
class_info = {
    'classes': class_names,
    'accuracy': float(test_acc),
    'loss': float(test_loss)
}

with open('classes.json', 'w') as f:
    json.dump(class_info, f)

print(f"✓ Class info saved: classes.json")

# Step 8: Plot results
print("\n" + "="*80)
print("STEP 8: PLOTTING RESULTS")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid()

# Loss
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.savefig('training_history.png')
print("✓ Training history saved: training_history.png")
plt.close()

print("\n" + "="*80)
print("✨ TRAINING COMPLETE!")
print("="*80)

print(f"\n📊 Results Summary:")
print(f"   • Model saved: {MODEL_NAME}")
print(f"   • Classes: {', '.join(class_names)}")
print(f"   • Test Accuracy: {test_acc*100:.2f}%")
print(f"   • Test Loss: {test_loss:.4f}")
print(f"   • Classes file: classes.json")
print(f"   • History plot: training_history.png")

print(f"\n🚀 Next step:")
print(f"   cd streamlit_app && streamlit run app.py")

print("\n" + "="*80)
