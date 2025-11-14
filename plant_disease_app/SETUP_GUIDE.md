# 🌿 Plant Disease Detection System - Setup Guide

## Environment Configuration ✓

Your Python virtual environment has been successfully configured with all required dependencies!

### Environment Details:
- **Type**: Virtual Environment (.venv)
- **Python Version**: 3.12.4
- **Location**: `C:\Users\hp\OneDrive\Desktop\capstone project\.venv`
- **Status**: ✅ All dependencies installed

### Installed Packages (Latest Versions):

#### Deep Learning & ML
- TensorFlow: 2.20.0
- Keras: 3.12.0
- NumPy: 2.2.6
- Pandas: 2.3.3
- Scikit-learn: 1.7.2
- Scikit-image: 0.25.2

#### Computer Vision
- OpenCV (cv2): 4.12.0.88
- Pillow: 12.0.0
- ImageIO: 2.37.2

#### Visualization
- Matplotlib: 3.10.7
- Seaborn: 0.13.2

#### Web Framework
- Streamlit: 1.51.0
- Streamlit-option-menu: 0.4.0

#### Utilities
- TQDM: 4.67.1
- PyYAML: 6.0.3
- Requests: 2.32.5

---

## ✨ Next Steps

### 1. **Prepare Your Dataset**

Organize your plant leaf images in the following structure:

```
plant_disease_app/data/raw_data/
├── Healthy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Powdery_Mildew/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Leaf_Spot/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── [Other_Disease_Classes]/
```

**Minimum Requirements:**
- At least 50 images per disease class
- 500+ images per class recommended for best results
- Supported formats: JPG, PNG, BMP, GIF
- Clear images with good lighting and focus

### 2. **Train the Model**

Navigate to the project directory and run the training script:

```bash
# Activate virtual environment (if not already active)
.venv\Scripts\Activate.ps1

# Run training with MobileNetV2 (recommended for first-time)
python src/train.py --data_dir data/raw_data --model MobileNetV2 --epochs 50

# Or try other architectures:
python src/train.py --data_dir data/raw_data --model ResNet50 --epochs 100
python src/train.py --data_dir data/raw_data --model EfficientNetB0 --epochs 100
```

**Training Parameters:**
- `--data_dir`: Path to your dataset folder
- `--model`: Model architecture (MobileNetV2, ResNet50, EfficientNetB0, CustomCNN)
- `--epochs`: Number of training epochs (default: 100)
- `--output_dir`: Where to save models (default: models/)

### 3. **Run the Web Application**

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Navigate to streamlit_app directory
cd streamlit_app

# Run the Streamlit app
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### 4. **Evaluate Model Performance**

The training script automatically generates:
- `{ModelName}_training_history.png` - Accuracy and loss curves
- `{ModelName}_confusion_matrix.png` - Confusion matrix heatmap
- `{ModelName}_results.json` - Model metrics and metadata

---

## 🔧 Troubleshooting

### Issue: Import errors when running scripts

**Solution:**
```bash
# Ensure virtual environment is activated
.venv\Scripts\Activate.ps1

# Verify all packages are installed
pip list

# Reinstall dependencies if needed
pip install -r requirements.txt
```

### Issue: Out of Memory (OOM) during training

**Solution:**
Reduce batch size in the training script or use a smaller model:

```bash
# Use MobileNetV2 with smaller epochs
python src/train.py --data_dir data/raw_data --model MobileNetV2 --epochs 30
```

### Issue: GPU not being used

**Solution:**
Check GPU availability with:

```python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If no GPU found and you have CUDA installed, install GPU support:

```bash
pip install tensorflow[and-cuda]
```

### Issue: Slow training on CPU

**Options:**
1. Reduce image size in `src/preprocess.py`: `img_size=160`
2. Use `CustomCNN` model (smaller and faster)
3. Install GPU support for faster training

---

## 📁 Project Structure Reminder

```
plant_disease_app/
├── data/                          # Your datasets
│   ├── raw_data/                 # Original images (organize by disease)
│   ├── train/                    # Training split
│   ├── val/                      # Validation split
│   └── test/                     # Test split
├── models/                        # Trained models & results
├── notebooks/                     # Jupyter notebooks
│   └── training_notebook.ipynb   # Full training pipeline
├── src/                          # Source code
│   ├── preprocess.py             # Data preprocessing
│   ├── model.py                  # Model architectures
│   ├── train.py                  # Training script
│   └── predict.py                # Inference module
├── streamlit_app/                # Web app
│   └── app.py                    # Main Streamlit app
├── requirements.txt              # Dependencies
└── README.md                     # Full documentation
```

---

## 🚀 Quick Commands Reference

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt

# Train model
python src/train.py --data_dir data/raw_data --model MobileNetV2 --epochs 50

# Run Streamlit app
cd streamlit_app && streamlit run app.py

# Check Python packages
pip list

# Deactivate virtual environment
deactivate
```

---

## ✅ Checklist

- [x] Virtual environment created and configured
- [x] All dependencies installed
- [x] Project folder structure ready
- [ ] Dataset prepared and organized
- [ ] Model trained
- [ ] Web app tested
- [ ] Ready for deployment

---

## 📞 Need Help?

1. **Check README.md** for comprehensive documentation
2. **Review training_notebook.ipynb** for step-by-step examples
3. **Check src/ modules** for detailed code documentation
4. **Verify data format** - ensure images are in correct folder structure

---

**Status**: ✅ Environment ready for training!  
**Next**: Prepare your dataset and run training script.

Good luck with your plant disease detection project! 🌿
