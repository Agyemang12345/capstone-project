# 🎯 RUN THIS COMMAND TO TRAIN YOUR MODEL

**Once pip finishes installing packages**, copy and paste this exact command in your PowerShell terminal:

---

## THE COMMAND

```powershell
python src/train.py --data_dir data/dataset/Train/Train --model MobileNetV2 --epochs 30
```

---

## WHAT THIS DOES

- ✅ Uses your dataset from `data/dataset/Train/Train/` folder
- ✅ Uses MobileNetV2 (fast, efficient model)
- ✅ Trains for 30 epochs (good balance of speed & accuracy)
- ✅ Saves trained model to `models/` folder
- ✅ Generates accuracy graphs
- ✅ Saves evaluation metrics

---

## EXPECTED OUTPUT

```
📊 Loading dataset from: data/dataset/Train/Train

Classes: ['Healthy', 'Powdery', 'Rust']
Total images: ~300

🤖 Creating MobileNetV2 model...

🎓 Starting training...

Epoch 1/30
200/200 [========>  ] - 45s 225ms/step - loss: 2.4532 - accuracy: 0.3456

Epoch 2/30  
200/200 [========>  ] - 40s 200ms/step - loss: 2.0123 - accuracy: 0.5234

... (continues)...

Epoch 30/30
200/200 [========>  ] - 40s 200ms/step - loss: 0.3456 - accuracy: 0.9234

✅ Training Complete!
Test Accuracy: 87.65%

✅ Model saved: models/model_MobileNetV2_20241114_143022.h5
✅ Graphs saved: models/training_history.png
✅ Metrics saved: models/model_MobileNetV2_20241114_143022_results.json
```

---

## ⏱️ TIME

**Total training time**: 15-30 minutes (depending on your computer)

---

## ✅ SUCCESS

After it finishes, you'll see files in `models/` folder:
- `model_MobileNetV2_*.h5` ← Your trained model!
- `training_history.png`
- `confusion_matrix.png`
- `*_results.json`

---

## 🚀 THEN RUN THE WEB APP

```powershell
cd streamlit_app
streamlit run app.py
```

Upload leaf images and get instant disease predictions! 🎉

---

**Ready? Just wait for pip to finish, then paste the command above!**

