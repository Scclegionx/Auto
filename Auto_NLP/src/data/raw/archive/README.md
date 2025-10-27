# Archive - Raw Data Files

This folder contains raw data files that have been processed and are no longer needed.

## Files Archived:
- elderly_command_dataset_bio_clean.json: Intermediate BIO dataset
- elderly_command_dataset_enriched.json: Enriched dataset (replaced by smart_augmented)
- elderly_command_dataset_final_bio.json: Final BIO dataset (replaced by clean_bio)
- elderly_command_dataset_smart_augmented.json: Smart augmented dataset (processed to clean_bio)
- elderly_command_dataset_unified.json: Original unified dataset (processed)

## Current Essential Files:
- elderly_command_dataset_clean_bio.json: Final clean dataset with BIO labels
- entity_vocab_clean.json: Clean entity vocabulary
- __init__.py: Python package initialization

## Dataset Processing Flow:
1. elderly_command_dataset_unified.json (original)
2. elderly_command_dataset_enriched.json (enriched)
3. elderly_command_dataset_smart_augmented.json (augmented)
4. elderly_command_dataset_clean_bio.json (final clean)

Archived on: 2025-10-20 17:39:08
