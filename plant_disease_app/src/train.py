"""
Training Script
Trains the plant disease classification model and saves the best model.
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import custom modules
from preprocess import ImagePreprocessor
from model import PlantDiseaseModel


class ModelTrainer:
    """
    Handles model training, evaluation, and checkpointing.
    """
    
    def __init__(self, model_name: str = "MobileNetV2", output_dir: str = "../models"):
        """
        Initialize trainer.
        
        Args:
            model_name: Name of the model architecture (MobileNetV2, ResNet50, EfficientNetB0, CustomCNN)
            output_dir: Directory to save models and results
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.history = None
        self.model = None
        self.class_names = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self, data_dir: str, val_split: float = 0.15, test_split: float = 0.15) -> dict:
        """
        Load and preprocess data from folder structure.
        
        Args:
            data_dir: Directory containing disease class folders
            val_split: Validation split ratio
            test_split: Test split ratio
            
        Returns:
            Dictionary containing train, val, and test datasets
        """
        print("Loading dataset...")
        preprocessor = ImagePreprocessor(img_size=224)
        
        # Load all images
        images, labels, class_names = preprocessor.load_dataset_from_folder(data_dir)
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Split dataset
        train_data, val_data, test_data = preprocessor.split_dataset(
            images, labels, 
            train_ratio=1 - val_split - test_split,
            val_ratio=val_split
        )
        
        # Create TensorFlow datasets
        train_dataset = preprocessor.prepare_tf_dataset(
            train_data[0], train_data[1], batch_size=32, augment=True
        )
        val_dataset = preprocessor.prepare_tf_dataset(
            val_data[0], val_data[1], batch_size=32, augment=False
        )
        test_dataset = preprocessor.prepare_tf_dataset(
            test_data[0], test_data[1], batch_size=32, augment=False
        )
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'test_images': test_data[0],
            'test_labels': test_data[1],
            'class_names': class_names
        }
    
    def create_model(self) -> tf.keras.Model:
        """
        Create the model based on specified architecture.
        
        Returns:
            Compiled Keras model
        """
        print(f"Creating {self.model_name} model...")
        model_factory = PlantDiseaseModel(num_classes=self.num_classes, img_size=224)
        
        if self.model_name == "MobileNetV2":
            model = model_factory.create_mobilenetv2()
        elif self.model_name == "ResNet50":
            model = model_factory.create_resnet50()
        elif self.model_name == "EfficientNetB0":
            model = model_factory.create_efficientnetb0()
        elif self.model_name == "CustomCNN":
            model = model_factory.create_custom_cnn()
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
        model_factory.get_model_info(model)
        self.model = model
        return model
    
    def get_callbacks(self, model_path: str) -> list:
        """
        Create training callbacks for model checkpointing and early stopping.
        
        Args:
            model_path: Path to save the best model
            
        Returns:
            List of Keras callbacks
        """
        return [
            # Save best model
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate scheduler
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.output_dir, 'logs'),
                histogram_freq=0,
                write_graph=True,
                update_freq='epoch'
            )
        ]
    
    def train(self, data_dict: dict, epochs: int = 100, use_augmentation: bool = True) -> None:
        """
        Train the model.
        
        Args:
            data_dict: Dictionary containing train, val, test datasets
            epochs: Number of training epochs
            use_augmentation: Whether to use data augmentation
        """
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        train_dataset = data_dict['train']
        val_dataset = data_dict['val']
        
        # Create model path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(
            self.output_dir, 
            f"{self.model_name}_{timestamp}.h5"
        )
        
        # Get callbacks
        callbacks = self.get_callbacks(model_path)
        
        # Train model
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        return model_path
    
    def evaluate(self, data_dict: dict) -> dict:
        """
        Evaluate model on test set.
        
        Args:
            data_dict: Dictionary containing test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL ON TEST SET")
        print("="*60)
        
        test_dataset = data_dict['test']
        test_images = data_dict['test_images']
        test_labels = data_dict['test_labels']
        
        # Evaluate on test set
        test_loss, test_accuracy, test_auc = self.model.evaluate(test_dataset, verbose=1)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        # Get predictions
        predictions = self.model.predict(test_images)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Classification report
        print("\n" + "-"*60)
        print("CLASSIFICATION REPORT")
        print("-"*60)
        print(classification_report(
            test_labels, predicted_labels,
            target_names=self.class_names,
            digits=4
        ))
        
        return {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'test_auc': float(test_auc),
            'predictions': predicted_labels,
            'true_labels': test_labels,
            'confidence_scores': predictions
        }
    
    def plot_training_history(self) -> None:
        """
        Plot and save training history (accuracy and loss curves).
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'{self.model_name} - Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title(f'{self.model_name} - Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'{self.model_name}_training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {plot_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self, eval_dict: dict) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            eval_dict: Dictionary containing evaluation results
        """
        predictions = eval_dict['predictions']
        true_labels = eval_dict['true_labels']
        
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cmap='Blues'
        )
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        cm_path = os.path.join(self.output_dir, f'{self.model_name}_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {cm_path}")
        
        plt.close()
    
    def save_results(self, eval_dict: dict, model_path: str) -> None:
        """
        Save training results to JSON file.
        
        Args:
            eval_dict: Dictionary containing evaluation results
            model_path: Path to the saved model
        """
        results = {
            'model_name': self.model_name,
            'model_path': model_path,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'test_accuracy': eval_dict['test_accuracy'],
            'test_loss': eval_dict['test_loss'],
            'test_auc': eval_dict['test_auc'],
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(self.output_dir, f'{self.model_name}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {results_path}")


def main():
    """
    Main training function with command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train plant disease detection model')
    
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Directory containing disease class folders'
    )
    parser.add_argument(
        '--model', type=str, default='MobileNetV2',
        choices=['MobileNetV2', 'ResNet50', 'EfficientNetB0', 'CustomCNN'],
        help='Model architecture to use'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--output_dir', type=str, default='../models',
        help='Directory to save models and results'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ModelTrainer(model_name=args.model, output_dir=args.output_dir)
    
    # Load data
    data_dict = trainer.load_data(args.data_dir)
    
    # Create model
    trainer.create_model()
    
    # Train model
    model_path = trainer.train(data_dict, epochs=args.epochs)
    
    # Evaluate model
    eval_dict = trainer.evaluate(data_dict)
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(eval_dict)
    
    # Save results
    trainer.save_results(eval_dict, model_path)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED!")
    print("="*60)
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
