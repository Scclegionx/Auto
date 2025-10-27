# Raw Data

This folder contains the essential raw data files for the Vietnamese NLP system.

## Current Files:

### Essential Datasets:
- **elderly_command_dataset_clean_bio.json**: Final clean dataset with BIO labels
  - 2,575 samples
  - Clean BIO format
  - Ready for training
- **entity_vocab_clean.json**: Clean entity vocabulary
  - 21 entity labels
  - Mapped to IDs
  - Used for training

### Dataset Statistics:
- Total samples: 2,575
- Intent distribution: call, send-mess, search-internet
- Entity types: CONTACT_NAME, MESSAGE, QUERY, TIME, etc.
- BIO format: Clean and validated

## Usage:

### For Training:
```bash
# The clean dataset is automatically used by training scripts
python src/training/scripts/train_gpu.py
```

### For Data Processing:
```bash
# Use management scripts to process data
python scripts/management/dataset_splitter.py
python scripts/management/smart_augmentation.py
python scripts/management/fix_bio_labels.py
```

## Archive:
- See `archive/` folder for processed intermediate files
- All essential data has been preserved

Last updated: 2025-10-20 17:39:08
