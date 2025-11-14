"""
Model Architecture Module
Defines CNN models for plant disease classification including MobileNetV2, ResNet50, and EfficientNetB0.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple


class PlantDiseaseModel:
    """
    Factory class for creating different CNN architectures for plant disease classification.
    """
    
    def __init__(self, num_classes: int, img_size: int = 224):
        """
        Initialize model creator.
        
        Args:
            num_classes: Number of disease classes
            img_size: Input image size (img_size x img_size)
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.input_shape = (img_size, img_size, 3)
    
    def create_mobilenetv2(self, learning_rate: float = 1e-3) -> models.Model:
        """
        Create MobileNetV2 model with transfer learning.
        
        MobileNetV2 is lightweight and efficient, ideal for mobile deployment.
        
        Args:
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create model architecture
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # Preprocessing layer for MobileNetV2
            tf.keras.applications.mobilenet_v2.preprocess_input,
            
            # Base model
            base_model,
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(256, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def create_resnet50(self, learning_rate: float = 1e-3) -> models.Model:
        """
        Create ResNet50 model with transfer learning.
        
        ResNet50 provides good balance between accuracy and computational efficiency.
        
        Args:
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained ResNet50
        base_model = tf.keras.applications.ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create model architecture
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # Preprocessing layer for ResNet50
            tf.keras.applications.resnet50.preprocess_input,
            
            # Base model
            base_model,
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers with batch normalization
            layers.Dense(512, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu', name='dense_3'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def create_efficientnetb0(self, learning_rate: float = 1e-3) -> models.Model:
        """
        Create EfficientNetB0 model with transfer learning.
        
        EfficientNetB0 is the smallest variant, optimized for efficiency.
        
        Args:
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained EfficientNetB0
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create model architecture
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # Preprocessing layer for EfficientNetB0
            tf.keras.applications.efficientnet.preprocess_input,
            
            # Base model
            base_model,
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(256, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def create_custom_cnn(self, learning_rate: float = 1e-3) -> models.Model:
        """
        Create a custom CNN from scratch for comparison.
        
        This model is smaller and can train faster but may have lower accuracy.
        
        Args:
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Preprocessing: normalize to [-1, 1]
            layers.Rescaling(1./127.5, offset=-1),
            
            # Conv Block 1
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            
            # Conv Block 2
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            
            # Conv Block 3
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.3),
            
            # Conv Block 4
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.3),
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def unfreeze_base_model(self, model: models.Model, num_layers_to_unfreeze: int = 50):
        """
        Unfreeze the last N layers of the base model for fine-tuning.
        
        This is useful after initial training to improve accuracy.
        
        Args:
            model: The compiled model
            num_layers_to_unfreeze: Number of layers to unfreeze from the end
        """
        # Find the base model layer
        base_model = None
        for layer in model.layers:
            if hasattr(layer, 'trainable') and len(layer.trainable_params) > 1000000:
                base_model = layer
                break
        
        if base_model is None:
            print("Could not find base model layer")
            return
        
        # Unfreeze layers
        base_model.trainable = True
        total_layers = len(base_model.layers)
        freeze_until = total_layers - num_layers_to_unfreeze
        
        for i, layer in enumerate(base_model.layers[:freeze_until]):
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        print(f"Unfroze last {num_layers_to_unfreeze} layers ({total_layers - freeze_until}/{total_layers})")
    
    def get_model_info(self, model: models.Model) -> None:
        """
        Print comprehensive model information.
        
        Args:
            model: Keras model
        """
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        print(f"Total parameters: {model.count_params():,}")
        
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_params = model.count_params() - trainable_params
        
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"Model size: ~{model.count_params() * 4 / 1024 / 1024:.2f} MB")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    num_classes = 5  # Number of disease classes
    
    model_factory = PlantDiseaseModel(num_classes=num_classes, img_size=224)
    
    # Create different models
    print("Creating MobileNetV2...")
    mobilenet = model_factory.create_mobilenetv2()
    model_factory.get_model_info(mobilenet)
    
    print("Creating ResNet50...")
    resnet = model_factory.create_resnet50()
    model_factory.get_model_info(resnet)
    
    print("Creating EfficientNetB0...")
    efficientnet = model_factory.create_efficientnetb0()
    model_factory.get_model_info(efficientnet)
    
    print("Creating Custom CNN...")
    custom_cnn = model_factory.create_custom_cnn()
    model_factory.get_model_info(custom_cnn)
