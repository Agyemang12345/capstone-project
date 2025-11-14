"""
Data Preprocessing Module
Handles image loading, augmentation, and dataset splitting for plant disease classification.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
from typing import Tuple, List


class ImagePreprocessor:
    """
    Handles all image preprocessing tasks including loading, augmentation, and normalization.
    """
    
    def __init__(self, img_size: int = 224):
        """
        Initialize the preprocessor.
        
        Args:
            img_size: Target size for all images (img_size x img_size)
        """
        self.img_size = img_size
        self.IMG_CHANNELS = 3
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def get_augmentation_transforms(self) -> tf.keras.preprocessing.image.ImageDataGenerator:
        """
        Create augmentation transforms for training data.
        
        Returns:
            ImageDataGenerator with augmentation parameters
        """
        return tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
    
    def load_dataset_from_folder(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all images from a folder structure where each subfolder is a disease class.
        
        Folder structure:
        data_dir/
          ├── Disease1/
          │   ├── image1.jpg
          │   └── image2.jpg
          ├── Disease2/
          │   ├── image3.jpg
          │   └── image4.jpg
        
        Args:
            data_dir: Root directory containing disease class folders
            
        Returns:
            Tuple of (images array, labels array, class names list)
        """
        images = []
        labels = []
        class_names = []
        
        # Get all subdirectories (disease classes)
        disease_dirs = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))])
        
        if not disease_dirs:
            raise ValueError(f"No subdirectories found in {data_dir}")
        
        class_names = disease_dirs
        
        # Load images from each disease folder
        for class_idx, disease_name in enumerate(disease_dirs):
            disease_path = os.path.join(data_dir, disease_name)
            
            # Get all image files
            image_files = [f for f in os.listdir(disease_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            
            print(f"Loading {len(image_files)} images from {disease_name}...")
            
            for img_file in image_files:
                img_path = os.path.join(disease_path, img_file)
                img = self.load_image(img_path)
                
                if img is not None:
                    images.append(img)
                    labels.append(class_idx)
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} images from {len(class_names)} classes")
        print(f"Image shape: {images.shape}")
        print(f"Class names: {class_names}")
        
        return images, labels, class_names
    
    def split_dataset(self, images: np.ndarray, labels: np.ndarray, 
                     train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            images: Array of preprocessed images
            labels: Array of corresponding labels
            train_ratio: Proportion for training (default: 70%)
            val_ratio: Proportion for validation (default: 15%)
            
        Returns:
            Tuple of (train_data, val_data, test_data) each containing (images, labels)
        """
        test_ratio = 1 - train_ratio - val_ratio
        
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_ratio, random_state=42, stratify=labels
        )
        
        # Second split: train vs val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape[0]} images")
        print(f"Val set: {X_val.shape[0]} images")
        print(f"Test set: {X_test.shape[0]} images")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def prepare_tf_dataset(self, images: np.ndarray, labels: np.ndarray, 
                           batch_size: int = 32, augment: bool = False) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from images and labels.
        
        Args:
            images: Array of preprocessed images
            labels: Array of corresponding labels
            batch_size: Batch size for training
            augment: Whether to apply data augmentation
            
        Returns:
            TensorFlow Dataset object
        """
        # Convert labels to one-hot encoding
        num_classes = len(np.unique(labels))
        labels_onehot = tf.keras.utils.to_categorical(labels, num_classes)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels_onehot))
        
        if augment:
            # Apply augmentation
            dataset = dataset.map(self._augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_image(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply random augmentation to a single image.
        
        Args:
            image: Input image tensor
            label: Corresponding label
            
        Returns:
            Augmented image and label
        """
        # Random rotation
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Random vertical flip
        image = tf.image.random_flip_up_down(image)
        
        # Random brightness adjustment
        image = tf.image.random_brightness(image, 0.2)
        
        # Ensure pixel values stay in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def save_split_dataset(self, data_dir: str, output_dir: str):
        """
        Load dataset from folder structure and save train/val/test splits to separate directories.
        
        This is useful for organizing data before training.
        
        Args:
            data_dir: Source directory with disease class folders
            output_dir: Output directory where splits will be saved
        """
        # Load full dataset
        images, labels, class_names = self.load_dataset_from_folder(data_dir)
        
        # Split dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split_dataset(images, labels)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            for disease in class_names:
                os.makedirs(os.path.join(output_dir, split, disease), exist_ok=True)
        
        # Save splits
        print("Saving train set...")
        self._save_split_images(X_train, y_train, class_names, os.path.join(output_dir, 'train'))
        
        print("Saving val set...")
        self._save_split_images(X_val, y_val, class_names, os.path.join(output_dir, 'val'))
        
        print("Saving test set...")
        self._save_split_images(X_test, y_test, class_names, os.path.join(output_dir, 'test'))
    
    def _save_split_images(self, images: np.ndarray, labels: np.ndarray, 
                          class_names: List[str], output_dir: str):
        """
        Save split images to folders organized by class.
        
        Args:
            images: Array of preprocessed images
            labels: Array of labels
            class_names: List of disease class names
            output_dir: Directory to save images
        """
        counters = {cls: 0 for cls in class_names}
        
        for img, label in zip(images, labels):
            disease_name = class_names[label]
            counter = counters[disease_name]
            
            # Convert image back to 0-255 range for saving
            img_uint8 = (img * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            
            # Save image
            output_path = os.path.join(output_dir, disease_name, f"{disease_name}_{counter}.jpg")
            cv2.imwrite(output_path, img_bgr)
            
            counters[disease_name] += 1


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor(img_size=224)
    
    # Assuming you have a folder structure like:
    # raw_data/
    #   ├── Healthy/
    #   ├── Powdery_Mildew/
    #   └── Leaf_Spot/
    
    # Load dataset
    # images, labels, class_names = preprocessor.load_dataset_from_folder("../data/raw_data")
    
    # Split dataset
    # train_data, val_data, test_data = preprocessor.split_dataset(images, labels)
    
    # Create TensorFlow datasets
    # train_dataset = preprocessor.prepare_tf_dataset(train_data[0], train_data[1], batch_size=32, augment=True)
    # val_dataset = preprocessor.prepare_tf_dataset(val_data[0], val_data[1], batch_size=32, augment=False)
    # test_dataset = preprocessor.prepare_tf_dataset(test_data[0], test_data[1], batch_size=32, augment=False)
