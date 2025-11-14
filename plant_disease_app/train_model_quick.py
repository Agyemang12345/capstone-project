#!/usr/bin/env python3
"""
Quick training script to train model using existing dataset
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import ModelTrainer

def main():
    """
    Main function to train the model
    """
    
    # Define paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "dataset"
    
    # Check if dataset exists
    if not data_dir.exists():
        print(f"❌ Dataset not found at: {data_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("🌿 PLANT DISEASE DETECTION - MODEL TRAINING")
    print("=" * 80)
    print(f"\n📁 Project Root: {project_root}")
    print(f"📁 Dataset Path: {data_dir}")
    print(f"\n🔍 Dataset Structure:")
    
    # Show dataset structure
    if (data_dir / "Train").exists():
        train_path = data_dir / "Train" / "Train"
        if train_path.exists():
            classes = [d.name for d in train_path.iterdir() if d.is_dir()]
            print(f"   Classes found: {', '.join(sorted(classes))}")
            for cls in sorted(classes):
                cls_path = train_path / cls
                num_images = len(list(cls_path.glob('*.*')))
                print(f"   - {cls}: {num_images} images")
    
    print("\n" + "=" * 80)
    print("🚀 STARTING TRAINING PROCESS")
    print("=" * 80)
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_name='MobileNetV2',
        data_dir=str(data_dir),
        epochs=30,
        batch_size=32,
        learning_rate=0.001,
        validation_split=0.2,
        test_split=0.1,
        augment_data=True
    )
    
    print("\n✅ Trainer initialized successfully!")
    print(f"   - Model: MobileNetV2 (Fast & Efficient)")
    print(f"   - Epochs: 30")
    print(f"   - Batch Size: 32")
    print(f"   - Data Augmentation: Enabled")
    
    print("\n📊 Loading dataset...")
    
    # Load data
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = trainer.load_data()
        print(f"✅ Dataset loaded successfully!")
        print(f"   - Classes: {len(class_names)} → {class_names}")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Validation samples: {len(X_val)}")
        print(f"   - Test samples: {len(X_test)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        sys.exit(1)
    
    print("\n🤖 Creating model...")
    
    # Create model
    try:
        model = trainer.create_model(len(class_names))
        print(f"✅ Model created successfully!")
        print(f"   - Architecture: MobileNetV2 with transfer learning")
        print(f"   - Input shape: (224, 224, 3)")
        print(f"   - Output classes: {len(class_names)}")
    except Exception as e:
        print(f"❌ Error creating model: {str(e)}")
        sys.exit(1)
    
    print("\n🎓 Training model (this may take a few minutes)...")
    print("-" * 80)
    
    # Train model
    try:
        history, model = trainer.train(X_train, y_train, X_val, y_val, model)
        print("-" * 80)
        print(f"✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        sys.exit(1)
    
    print("\n📈 Evaluating model...")
    
    # Evaluate
    try:
        test_loss, test_accuracy, test_report = trainer.evaluate(X_test, y_test, model, class_names)
        print(f"✅ Model evaluation completed!")
        print(f"   - Test Accuracy: {test_accuracy:.2%}")
        print(f"   - Test Loss: {test_loss:.4f}")
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        sys.exit(1)
    
    print("\n📊 Plotting results...")
    
    # Plot results
    try:
        trainer.plot_training_history(history)
        trainer.plot_confusion_matrix(y_test, model.predict(X_test), class_names)
        print(f"✅ Plots generated successfully!")
        print(f"   - Saved: training_history.png")
        print(f"   - Saved: confusion_matrix.png")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate plots: {str(e)}")
    
    print("\n💾 Saving results...")
    
    # Save results
    try:
        results_file = trainer.save_results(test_loss, test_accuracy, test_report, class_names)
        print(f"✅ Results saved successfully!")
        print(f"   - File: {results_file}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save results file: {str(e)}")
    
    print("\n" + "=" * 80)
    print("✨ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n🎉 Your trained model is ready!")
    print(f"📁 Location: {project_root / 'models' / 'model_MobileNetV2*.h5'}")
    print(f"\n🚀 Next steps:")
    print(f"   1. Go to streamlit_app folder")
    print(f"   2. Run: streamlit run app.py")
    print(f"   3. Upload a leaf image to test predictions!")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
