# 📦 MODEL SETUP GUIDE

## ⚠️ Model Files Not Included in Git

**Reason**: Trained model file (`best_model.pt`) is **1.4GB** - exceeds Git's 100MB file limit.

---

## 🎯 OPTION 1: Download Pre-trained Model (Recommended)

### For Team Members / Deployment

**Model Location**: 
- Upload `best_model.pt` to cloud storage (Google Drive, Dropbox, S3, etc.)
- Share download link with team

**Download Instructions**:
```bash
# Example: Download from Google Drive
# Replace FILE_ID with your actual file ID
curl -L "https://drive.google.com/uc?export=download&id=FILE_ID" -o models/phobert_multitask_lr1e5/best_model.pt

# Or use wget
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILE_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILE_ID" -O models/phobert_multitask_lr1e5/best_model.pt && rm -rf /tmp/cookies.txt
```

### Required Model Structure
After download, verify this structure:
```
models/
└── phobert_multitask_lr1e5/
    ├── best_model.pt          # 1.4GB - DOWNLOAD THIS
    ├── checkpoint-epoch1.pt   # (optional backup)
    ├── config.json            # ✅ In Git
    ├── training_config.json   # ✅ In Git
    └── ... (other configs)
```

---

## 🎯 OPTION 2: Use Git LFS (Large File Storage)

If you want to track model files in Git:

### 1. Install Git LFS
```bash
# Windows (with Git)
git lfs install

# Linux
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

# macOS
brew install git-lfs
git lfs install
```

### 2. Track Model Files
```bash
# Track .pt files with LFS
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes

# Add and commit model
git add models/phobert_multitask_lr1e5/best_model.pt
git commit -m "Add trained model with LFS"
git push origin main
```

### 3. Clone with LFS
Team members need to:
```bash
git lfs install
git clone <repo-url>
git lfs pull
```

**Note**: GitHub LFS has bandwidth limits (1GB/month free)

---

## 🎯 OPTION 3: Retrain from Scratch

If you have the training data:

```bash
# Run training script
cd Auto_NLP
python src/training/train_multitask.py \
  --data_dir src/data/processed \
  --output_dir models/phobert_multitask_lr1e5 \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 1e-5
```

**Training time**: ~4-8 hours on GPU, ~2-3 days on CPU

---

## ✅ Verify Model Installation

After obtaining the model file:

```bash
# Check file exists
ls -lh models/phobert_multitask_lr1e5/best_model.pt

# Expected output:
# -rw-r--r-- 1 user user 1.4G Dec 27 22:00 best_model.pt

# Test API startup
python api/server.py
```

If successful, you should see:
```
✅ Loaded checkpoint from models\phobert_multitask\best_model.pt
✅ Trained model loaded successfully
```

---

## 🚨 Troubleshooting

### Issue: Model not found
```
FileNotFoundError: models/phobert_multitask_lr1e5/best_model.pt
```

**Solution**: 
1. Verify file exists in correct location
2. Check file permissions
3. Download model using instructions above

### Issue: Model loading error
```
RuntimeError: Error loading model checkpoint
```

**Solution**:
1. Verify file is complete (1.4GB, not corrupted)
2. Check PyTorch version compatibility
3. Re-download if necessary

---

## 📊 Model Information

**Model**: PhoBERT Multi-task (Intent Classification + NER)
- **Architecture**: RoBERTa-based Vietnamese BERT
- **Base Model**: vinai/phobert-large
- **Tasks**: 
  - Intent Classification (11 intents)
  - Named Entity Recognition (BIO tagging)
- **Training Data**: ~15,000 Vietnamese voice commands
- **Performance**: 
  - Intent Accuracy: 95%+
  - Entity F1: 88%+

---

## 📞 Support

For model access or training issues, contact:
- [Your contact info]
- [Alternative download link]




