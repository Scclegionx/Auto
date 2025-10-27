"""
Dataset Processor for Imbalanced Dataset
Xử lý dataset với class imbalance và data augmentation
"""

import json
import random
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
import numpy as np

class ImbalancedDatasetProcessor:
    """Processor cho dataset bị imbalanced"""
    
    def __init__(self, tokenizer_name: str = "vinai/phobert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.intent_mapping = {
            'send-mess': 0, 'call': 1, 'set-alarm': 2, 'set-event-calendar': 3,
            'get-info': 4, 'add-contacts': 5, 'control-device': 6, 'make-video-call': 7,
            'open-cam': 8, 'play-media': 9, 'search-internet': 10, 'view-content': 11,
            'search-youtube': 12
        }
        self.id_to_intent = {v: k for k, v in self.intent_mapping.items()}
    
    def load_dataset(self, file_path: str) -> List[Dict]:
        """Load dataset từ file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_class_distribution(self, data: List[Dict]) -> Dict[str, int]:
        """Phân tích phân phối class"""
        class_counts = Counter()
        for item in data:
            intent = item.get('command', 'unknown')
            class_counts[intent] += 1
        
        print("Class distribution:")
        for intent, count in class_counts.most_common():
            percentage = (count / len(data)) * 100
            print(f"  {intent}: {count} ({percentage:.1f}%)")
        
        return dict(class_counts)
    
    def augment_minority_classes(self, data: List[Dict], target_samples: int = 1000) -> List[Dict]:
        """Augment minority classes để cân bằng dataset"""
        class_counts = Counter(item.get('command', 'unknown') for item in data)
        max_count = max(class_counts.values())
        
        augmented_data = data.copy()
        
        for intent, count in class_counts.items():
            if count < target_samples:
                # Get samples of this class
                class_samples = [item for item in data if item.get('command') == intent]
                
                # Generate augmented samples
                needed_samples = target_samples - count
                for _ in range(needed_samples):
                    # Random augmentation strategies
                    sample = random.choice(class_samples).copy()
                    
                    # Strategy 1: Synonym replacement
                    if random.random() < 0.3:
                        sample['input'] = self._replace_synonyms(sample['input'])
                    
                    # Strategy 2: Word order variation
                    elif random.random() < 0.3:
                        sample['input'] = self._vary_word_order(sample['input'])
                    
                    # Strategy 3: Add filler words
                    elif random.random() < 0.3:
                        sample['input'] = self._add_filler_words(sample['input'])
                    
                    # Strategy 4: Paraphrase
                    else:
                        sample['input'] = self._paraphrase(sample['input'])
                    
                    augmented_data.append(sample)
        
        return augmented_data
    
    def _replace_synonyms(self, text: str) -> str:
        """Replace synonyms trong text"""
        synonyms = {
            'gọi': ['gọi', 'gọi điện', 'gọi cho'],
            'nhắn': ['nhắn', 'gửi tin', 'gửi tin nhắn'],
            'tìm': ['tìm', 'tìm kiếm', 'search'],
            'bật': ['bật', 'mở', 'khởi động'],
            'tắt': ['tắt', 'đóng', 'dừng']
        }
        
        words = text.split()
        for i, word in enumerate(words):
            for original, replacements in synonyms.items():
                if original in word.lower():
                    words[i] = random.choice(replacements)
                    break
        
        return ' '.join(words)
    
    def _vary_word_order(self, text: str) -> str:
        """Vary word order trong text"""
        words = text.split()
        if len(words) > 3:
            # Swap adjacent words
            i = random.randint(0, len(words) - 2)
            words[i], words[i + 1] = words[i + 1], words[i]
        return ' '.join(words)
    
    def _add_filler_words(self, text: str) -> str:
        """Add filler words"""
        fillers = ['nha', 'nhé', 'dùm', 'giùm', 'cái']
        filler = random.choice(fillers)
        
        # Insert filler at random position
        words = text.split()
        if len(words) > 1:
            pos = random.randint(0, len(words))
            words.insert(pos, filler)
        
        return ' '.join(words)
    
    def _paraphrase(self, text: str) -> str:
        """Paraphrase text"""
        paraphrases = {
            'gọi cho': ['gọi điện cho', 'gọi cho', 'liên lạc với'],
            'nhắn tin': ['gửi tin nhắn', 'nhắn tin', 'gửi tin'],
            'tìm kiếm': ['tìm', 'search', 'tìm kiếm'],
            'bật': ['mở', 'khởi động', 'bật'],
            'tắt': ['đóng', 'dừng', 'tắt']
        }
        
        result = text
        for original, replacements in paraphrases.items():
            if original in result:
                result = result.replace(original, random.choice(replacements))
        
        return result
    
    def create_weighted_sampler(self, data: List[Dict]) -> WeightedRandomSampler:
        """Tạo weighted sampler cho imbalanced dataset"""
        class_counts = Counter(item.get('command', 'unknown') for item in data)
        max_count = max(class_counts.values())
        
        # Calculate weights (inverse frequency)
        weights = []
        for item in data:
            intent = item.get('command', 'unknown')
            weight = max_count / class_counts[intent]
            weights.append(weight)
        
        return WeightedRandomSampler(weights, len(weights))
    
    def split_dataset(self, data: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset với stratification"""
        from sklearn.model_selection import train_test_split
        
        # Extract labels for stratification
        labels = [item.get('command', 'unknown') for item in data]
        
        train_data, val_data = train_test_split(
            data,
            test_size=1 - train_ratio,
            random_state=42,
            stratify=labels
        )
        
        return train_data, val_data
    
    def create_dataloader(self, data: List[Dict], batch_size: int = 16, 
                         shuffle: bool = True, use_weighted_sampling: bool = False) -> DataLoader:
        """Tạo DataLoader với options"""
        
        if use_weighted_sampling:
            sampler = self.create_weighted_sampler(data)
            return DataLoader(
                data,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=self.collate_fn
            )
        else:
            return DataLoader(
                data,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=self.collate_fn
            )
    
    def collate_fn(self, batch):
        """Collate function cho DataLoader"""
        texts = [item['input'] for item in batch]
        intents = [item.get('command', 'unknown') for item in batch]
        
        # Tokenize
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,  # Optimized for dataset
            return_tensors='pt'
        )
        
        # Convert intents to IDs
        intent_ids = []
        for intent in intents:
            intent_ids.append(self.intent_mapping.get(intent, 0))
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'intent_labels': torch.tensor(intent_ids)
        }
    
    def process_dataset(self, file_path: str, augment: bool = True, 
                       target_samples: int = 1000) -> Tuple[List[Dict], List[Dict]]:
        """Process dataset hoàn chỉnh"""
        print("Loading dataset...")
        data = self.load_dataset(file_path)
        
        print(f"Original dataset size: {len(data)}")
        
        # Analyze class distribution
        class_counts = self.analyze_class_distribution(data)
        
        # Augment minority classes if requested
        if augment:
            print("Augmenting minority classes...")
            data = self.augment_minority_classes(data, target_samples)
            print(f"Augmented dataset size: {len(data)}")
            
            # Re-analyze after augmentation
            self.analyze_class_distribution(data)
        
        # Split dataset
        train_data, val_data = self.split_dataset(data)
        
        print(f"Train: {len(train_data)} samples")
        print(f"Validation: {len(val_data)} samples")
        
        return train_data, val_data

def main():
    """Test dataset processor"""
    processor = ImbalancedDatasetProcessor()
    
    # Process dataset
    train_data, val_data = processor.process_dataset(
        "src/data/raw/elderly_command_dataset_FULL_13C_FIXED.json",
        augment=True,
        target_samples=1000
    )
    
    # Create dataloaders
    train_loader = processor.create_dataloader(
        train_data, 
        batch_size=16, 
        use_weighted_sampling=True
    )
    
    val_loader = processor.create_dataloader(
        val_data, 
        batch_size=16, 
        shuffle=False
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Test batch
    for batch in train_loader:
        print(f"Batch shape: {batch['input_ids'].shape}")
        print(f"Labels: {batch['intent_labels']}")
        break

if __name__ == "__main__":
    main()

