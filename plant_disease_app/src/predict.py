"""
Prediction Module
Loads trained model and performs inference on plant images.
"""

import os
import json
import numpy as np
import tensorflow as tf
import cv2
from typing import Tuple, Dict
from pathlib import Path


class PlantDiseasePredictior:
    """
    Load trained model and perform inference with confidence scoring.
    """
    
    def __init__(self, model_path: str, img_size: int = 224):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to the trained model (.h5 or .pth)
            img_size: Input image size
        """
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.class_names = None
        self.num_classes = None
        
        # Load model
        self.load_model()
    
    def load_model(self) -> None:
        """
        Load the trained model from disk.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded successfully")
        
        # Print model summary
        print(self.model.summary())
    
    def load_class_names(self, class_names_path: str) -> None:
        """
        Load class names from JSON file.
        
        Args:
            class_names_path: Path to JSON file containing class names
        """
        if not os.path.exists(class_names_path):
            raise FileNotFoundError(f"Class names file not found at {class_names_path}")
        
        with open(class_names_path, 'r') as f:
            data = json.load(f)
            self.class_names = data.get('class_names', [])
            self.num_classes = len(self.class_names)
        
        print(f"Loaded {self.num_classes} classes: {self.class_names}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image for prediction.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def predict_single(self, image_path: str) -> Dict[str, any]:
        """
        Predict disease on a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess image
        img = self.preprocess_image(image_path)
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_batch, verbose=0)[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[::-1][:3]
        top_predictions = [
            {
                'class': self.class_names[idx],
                'confidence': float(predictions[idx])
            }
            for idx in top_indices
        ]
        
        result = {
            'image_path': image_path,
            'predicted_disease': predicted_class,
            'confidence': confidence,
            'all_predictions': predictions.tolist(),
            'top_3_predictions': top_predictions
        }
        
        return result
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Predict on multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error predicting for {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results


def load_predictor_with_config(config_dir: str) -> PlantDiseasePredictior:
    """
    Load predictor using configuration from directory.
    
    Directory should contain:
    - model.h5 (trained model)
    - model_config.json (metadata with class names)
    
    Args:
        config_dir: Directory containing model files
        
    Returns:
        Initialized PlantDiseasePredictior
    """
    model_path = os.path.join(config_dir, 'model.h5')
    config_path = os.path.join(config_dir, 'model_config.json')
    
    predictor = PlantDiseasePredictior(model_path)
    predictor.load_class_names(config_path)
    
    return predictor


# Disease information database
DISEASE_INFO = {
    'Healthy': {
        'description': 'The leaf is healthy with no visible diseases.',
        'symptoms': 'Normal green color, no spots or discoloration',
        'solutions': ['Continue regular plant care', 'Maintain proper watering', 'Ensure adequate sunlight']
    },
    'Powdery_Mildew': {
        'description': 'Fungal infection causing white powdery coating on leaves.',
        'symptoms': 'White, powdery spots on leaves, stems, and flowers',
        'solutions': [
            'Remove infected leaves',
            'Improve air circulation',
            'Apply sulfur or neem oil fungicide',
            'Avoid overhead watering'
        ]
    },
    'Leaf_Spot': {
        'description': 'Fungal or bacterial disease causing brown spots on leaves.',
        'symptoms': 'Brown or black circular spots with yellow halo',
        'solutions': [
            'Remove infected leaves',
            'Apply copper fungicide',
            'Avoid wetting foliage',
            'Space plants for air circulation',
            'Sanitize tools and containers'
        ]
    },
    'Rust': {
        'description': 'Fungal disease causing orange/brown rusty spots.',
        'symptoms': 'Orange, brown, or reddish pustules on leaf undersides',
        'solutions': [
            'Remove infected leaves',
            'Apply sulfur-based fungicide',
            'Reduce leaf wetness',
            'Improve air circulation',
            'Remove infected debris'
        ]
    },
    'Blight': {
        'description': 'Serious fungal disease causing rapid plant decline.',
        'symptoms': 'Water-soaked spots, rapid leaf blackening, stem rot',
        'solutions': [
            'Remove infected plant parts',
            'Apply copper or chlorothalonil fungicide',
            'Improve drainage',
            'Avoid overhead watering',
            'Consider removing severely affected plants',
            'Ensure adequate spacing'
        ]
    },
    'Late_Blight': {
        'description': 'Serious oomycete disease affecting leaves, stems, and tubers. Rapid progression in cool, wet conditions.',
        'symptoms': 'Water-soaked spots on leaves, white fungal growth on undersides, blackened stems, rapid plant decline',
        'solutions': [
            'Remove infected leaves and stems immediately',
            'Apply copper or chlorothalonil fungicide',
            'Improve drainage and air circulation',
            'Avoid overhead watering (water at soil level)',
            'Remove and destroy infected plant material',
            'Apply fungicide preventatively in wet conditions',
            'Space plants for good air flow',
            'Consider crop rotation'
        ]
    }
}


# Aliases to map model class names to DISEASE_INFO keys
# Model outputs: 'Healthy', 'Powdery', 'Rust', 'Leaf_Spot', 'Late_Blight'
# Need to map to: 'Healthy', 'Powdery_Mildew', 'Rust', 'Leaf_Spot', 'Late_Blight'
DISEASE_NAME_ALIASES = {
    'healthy': 'Healthy',
    'powdery': 'Powdery_Mildew',
    'powdery_mildew': 'Powdery_Mildew',
    'powdery mildew': 'Powdery_Mildew',
    'rust': 'Rust',
    'leaf_spot': 'Leaf_Spot',
    'leaf spot': 'Leaf_Spot',
    'blight': 'Blight',
    'late_blight': 'Late_Blight',
    'late blight': 'Late_Blight'
}


def get_disease_info(disease_name: str) -> Dict[str, any]:
    """
    Get disease information for a predicted disease.

    This function maps model class names to standardized disease information.
    It uses a multi-step approach:
    1. Try direct match in DISEASE_INFO
    2. Normalize and try alias mapping
    3. Try case-insensitive matching
    4. Return fallback if no match found

    Args:
        disease_name: Name of the disease (as returned by the model)

    Returns:
        Dictionary with disease description, symptoms, and solutions
    """
    if not disease_name:
        return {
            'description': 'Information not available',
            'symptoms': 'Please consult with a plant pathologist',
            'solutions': ['Monitor the plant carefully', 'Take preventive measures']
        }

    # Step 1: Try direct match first
    if disease_name in DISEASE_INFO:
        return DISEASE_INFO[disease_name]

    # Step 2: Normalize and try aliases
    normalized = disease_name.strip().lower().replace('-', '_')
    
    # Check if normalized version is in aliases
    if normalized in DISEASE_NAME_ALIASES:
        mapped_name = DISEASE_NAME_ALIASES[normalized]
        if mapped_name in DISEASE_INFO:
            return DISEASE_INFO[mapped_name]
    
    # Step 3: Try case-insensitive comparison with DISEASE_INFO keys
    normalized_underscore = normalized.replace(' ', '_')
    for key in DISEASE_INFO:
        if key.lower() == normalized_underscore or key.lower().replace('_', '') == normalized.replace('_', ''):
            return DISEASE_INFO[key]
    
    # Step 4: Fallback with default safe information
    print(f"Warning: Disease '{disease_name}' not found in database. Using fallback.")
    return {
        'description': 'Information not available',
        'symptoms': 'Please consult with a plant pathologist',
        'solutions': ['Monitor the plant carefully', 'Take preventive measures']
    }


if __name__ == "__main__":
    # Example usage
    model_path = "../models/model.h5"
    config_dir = "../models"
    
    # Load predictor
    predictor = load_predictor_with_config(config_dir)
    
    # Example prediction
    # image_path = "path/to/test_image.jpg"
    # result = predictor.predict_single(image_path)
    # print(result)
    # 
    # # Get disease info
    # disease_info = get_disease_info(result['predicted_disease'])
    # print(disease_info)
