# 🚀 TRAINING YOUR MODEL - STEP-BY-STEP GUIDE

**Status**: Installation in progress → Ready for training

---

## ⏳ WHAT'S HAPPENING NOW

The pip installation is currently downloading and installing all 40+ packages (including TensorFlow 331.9 MB, which takes time).

**You will see in your terminal:**
```
Installing collected packages: pytz, namex, libclang, flatbuffers, ... 
[============================>  ] 75% Complete
```

---

## 🎯 WHAT TO DO NOW

### Step 1: Wait for Pip Installation (In Your Terminal)

In the PowerShell window where you activated `.venv`, **WAIT** until you see:
```
Successfully installed tensorflow keras numpy ... and 50+ more packages
```

⏱️ **Expected time**: 5-15 minutes (depending on your internet and disk speed)

---

## Step 2: After Installation - Train the Model

Once pip finishes, run this exact command in your activated terminal:

```powershell
python src/train.py --data_dir data/dataset/Train/Train --model MobileNetV2 --epochs 30
```

---

## 📊 WHAT YOU'LL SEE DURING TRAINING

### Phase 1: Data Loading (30 seconds - 2 minutes)
```
Loading dataset from: data/dataset/Train/Train
Found classes: ['Healthy', 'Powdery', 'Rust']
Loading images...
✓ Loaded XXX Healthy images
✓ Loaded XXX Powdery images  
✓ Loaded XXX Rust images
```

### Phase 2: Model Creation (30 seconds)
```
Creating MobileNetV2 model...
Loading ImageNet weights...
✓ Model created successfully
```

### Phase 3: Training (5-15 minutes)
```
Epoch 1/30
200/200 [========================================] - 45s 225ms/step 
loss: 2.4532 - accuracy: 0.3456
val_loss: 2.1234 - val_accuracy: 0.5678

Epoch 2/30
200/200 [========================================] - 40s 200ms/step
loss: 2.0123 - accuracy: 0.5234
val_loss: 1.8765 - val_accuracy: 0.6234

... (continues through 30 epochs) ...

Epoch 30/30
200/200 [========================================] - 40s 200ms/step
loss: 0.3456 - accuracy: 0.9234
val_loss: 0.4567 - val_accuracy: 0.8901
```

### Phase 4: Evaluation (1-2 minutes)
```
Evaluating model on test set...
Test Accuracy: 87.65%
Test Loss: 0.4523

Classification Report:
              precision   recall  f1-score  support
     Healthy      0.92      0.88      0.90      50
    Powdery       0.85      0.89      0.87      45
        Rust      0.88      0.90      0.89      52
```

### Phase 5: Results & Graphs (30 seconds)
```
Saving trained model...
✓ Model saved: models/model_MobileNetV2_20241114_143022.h5

Plotting training history...
✓ Graph saved: models/training_history.png

Creating confusion matrix...
✓ Graph saved: models/confusion_matrix.png

Saving results to JSON...
✓ Results saved: models/model_MobileNetV2_20241114_143022_results.json
```

---

## ✅ SUCCESS CRITERIA

After training completes, you should see:

✅ **In `models/` folder:**
- `model_MobileNetV2_*.h5` (your trained model - ~50-100 MB)
- `training_history.png` (accuracy/loss curves)
- `confusion_matrix.png` (prediction accuracy per class)
- `model_MobileNetV2_*_results.json` (detailed metrics)

✅ **Model Performance:**
- Test Accuracy > 80%
- Validation Accuracy > 85%
- Low validation loss (< 0.5)

✅ **Terminal Output:**
- No errors
- Clear completion message
- All files saved successfully

---

## 🎮 AFTER TRAINING SUCCEEDS

### Then Run the Web App:

```powershell
cd streamlit_app
streamlit run app.py
```

You'll see:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

  Press CTRL+C to stop the server.
```

**Your browser will open automatically** showing the web app!

---

## 📁 YOUR DATASET STRUCTURE

Training will automatically find:
```
data/dataset/Train/Train/
├── Healthy/          ← Healthy leaves
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (all healthy images)
├── Powdery/          ← Powdery mildew disease
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (all powdery images)
└── Rust/             ← Rust disease
    ├── image1.jpg
    ├── image2.jpg
    └── ... (all rust images)
```

The script will:
1. Load all images
2. Automatically resize to 224×224
3. Normalize pixel values
4. Split into train/validation/test
5. Apply data augmentation
6. Train the model

---

## 🔧 TRAINING CONFIGURATION

| Setting | Value | Why |
|---------|-------|-----|
| Model | MobileNetV2 | Fast (~5-10 min), mobile-friendly, 85-90% accuracy |
| Epochs | 30 | Balances training time and accuracy |
| Batch Size | 32 | Standard for most hardware |
| Validation Split | 20% | Standard practice |
| Data Augmentation | Yes | Improves generalization |

---

## 🚨 IF SOMETHING GOES WRONG

### Error: "ModuleNotFoundError: No module named 'tensorflow'"
```bash
# Make sure .venv is activated, then:
pip install tensorflow==2.20.0
```

### Error: "No images found"
```bash
# Check dataset exists:
ls data/dataset/Train/Train/
# Should show: Healthy  Powdery  Rust
```

### Error: "GPU not available" (This is OK!)
```
The script will automatically use CPU instead - just slower training
```

### Training very slow?
- You're using CPU (normal - takes longer)
- Reduce epochs: `--epochs 15`
- Reduce batch size: `--batch_size 16`

---

## 📞 QUICK COMMANDS REFERENCE

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Train model (MAIN COMMAND)
python src/train.py --data_dir data/dataset/Train/Train --model MobileNetV2 --epochs 30

# Alternative models (if MobileNetV2 is too slow):
python src/train.py --data_dir data/dataset/Train/Train --model ResNet50 --epochs 20
python src/train.py --data_dir data/dataset/Train/Train --model EfficientNetB0 --epochs 20

# Run web app
cd streamlit_app
streamlit run app.py

# List installed packages
pip list

# Check if TensorFlow works
python -c "import tensorflow; print(tensorflow.__version__)"
```

---

## ⏱️ COMPLETE TIMELINE

| Step | Time | What Happens |
|------|------|-------------|
| 1. pip installs packages | 5-15 min | Terminal shows "Installing..." |
| 2. Python loads dataset | 1-2 min | Reads images from folders |
| 3. Model trains | 5-15 min | Epoch 1/30, 2/30, ... 30/30 |
| 4. Model evaluates | 1-2 min | Tests on validation data |
| 5. Saves results | 30 sec | Saves model + graphs |
| **TOTAL** | **15-35 min** | **Model ready!** |

---

## 🎯 YOUR NEXT EXACT STEPS

### NOW:
1. **In your PowerShell terminal**, watch for `Successfully installed...`
2. **Wait patiently** (pip is downloading ~500 MB of packages)

### WHEN YOU SEE "Successfully installed":
3. **Copy & Paste** this command:
   ```powershell
   python src/train.py --data_dir data/dataset/Train/Train --model MobileNetV2 --epochs 30
   ```
4. **Press Enter** and watch the magic happen! 🎉

### WHEN TRAINING COMPLETES:
5. **Check** `models/` folder for `.h5` file (should be there!)
6. **Run** the web app:
   ```powershell
   cd streamlit_app && streamlit run app.py
   ```
7. **Upload leaf images** and test predictions!

---

## 🎉 YOU'RE READY!

Everything is set up. Just **wait for pip, then run the training command**!

**Estimated time to working disease detector: ~30-40 minutes total**

---

**Need help?** Check the dataset structure above or re-read the training config section.

**Let's build a plant disease detector!** 🌿🤖

