#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cập nhật dataset chính với command-entity mapping đúng và IOB2 labels
"""

import json
import re
from pathlib import Path
from collections import defaultdict

def update_main_dataset():
    """Cập nhật dataset chính với mapping đúng"""
    
    print("🔄 UPDATING MAIN DATASET")
    print("=" * 60)
    
    # Yêu cầu command-entity mapping
    REQUIRED_MAPPING = {
        "add-contacts": ["CONTACT_NAME", "PHONE"],
        "call": ["CONTACT_NAME", "PHONE", "RECEIVER", "PLATFORM"],
        "make-video-call": ["CONTACT_NAME", "RECEIVER", "PLATFORM"],
        "send-mess": ["RECEIVER", "MESSAGE", "PLATFORM"],
        "set-alarm": ["TIME", "DATE"],
        "set-event-calendar": ["TITLE", "DATE"],
        "play-media": ["MEDIA_TYPE"],
        "view-content": ["CONTENT_TYPE"],
        "search-internet": ["PLATFORM", "QUERY"],
        "search-youtube": ["PLATFORM", "QUERY"],
        "get-info": ["QUERY", "CONTENT_TYPE", "PLATFORM"],
        "control-device": ["ACTION", "DEVICE", "MODE"],
        "open-cam": ["ACTION", "MODE", "CAMERA_TYPE"]
    }
    
    # Load dataset chính
    main_dataset_path = "src/data/raw/elderly_command_dataset_MERGED_13C_VITEXT.json"
    print(f"📖 Loading main dataset from {main_dataset_path}...")
    
    with open(main_dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Patterns để extract missing entities
    EXTRACTION_PATTERNS = {
        "PLATFORM": [
            r"(zalo|viber|skype|facetime|messenger|telegram|whatsapp|google|bing|youtube|sms|tin nhắn|text)",
            r"(qua|trên|bằng)\s+(zalo|viber|skype|facetime|messenger|telegram|whatsapp|google|bing|youtube|sms|tin nhắn|text)"
        ],
        "ACTION": [
            r"(bật|tắt|mở|đóng|tăng|giảm|on|off|up|down)",
            r"(mở|bật|tắt|đóng)\s+(camera|cam|máy ảnh)"
        ],
        "DATE": [
            r"(hôm nay|mai|ngày mai|hôm qua)",
            r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})",
            r"(\d{1,2})\s+(tháng|month)\s+(\d{4})",
            r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})",
            r"(ngày|tháng|năm)\s+(\d+)"
        ],
        "TITLE": [
            r"(?:họp|cuộc họp|sự kiện|lịch|nhắc nhở|ghi nhớ)\s+([A-Za-zÀ-ỹ\s]+?)(?:\s+ngày|\s+lúc|\s+tại|$)",
            r"(?:đặt lịch|tạo lịch|nhắc tôi)\s+([A-Za-zÀ-ỹ\s]+?)(?:\s+ngày|\s+lúc|\s+tại|$)"
        ]
    }
    
    def extract_missing_entities(text, command, existing_entities):
        """Extract missing entities dựa trên patterns"""
        required_entities = REQUIRED_MAPPING.get(command, [])
        enhanced_entities = existing_entities.copy()
        
        for entity_type in required_entities:
            if entity_type not in enhanced_entities or not enhanced_entities[entity_type]:
                if entity_type in EXTRACTION_PATTERNS:
                    for pattern in EXTRACTION_PATTERNS[entity_type]:
                        matches = re.finditer(pattern, text, re.IGNORECASE)
                        for match in matches:
                            entity_text = match.group().strip()
                            if entity_text and len(entity_text) > 1:
                                enhanced_entities[entity_type] = entity_text
                                break
                        if entity_type in enhanced_entities and enhanced_entities[entity_type]:
                            break
        
        return enhanced_entities
    
    def create_entity_from_text(text, entity_text, entity_type, start_pos):
        """Tạo entity object từ text"""
        # Tìm vị trí thực tế của entity trong text
        entity_lower = entity_text.lower()
        text_lower = text.lower()
        
        # Tìm vị trí bắt đầu
        actual_start = text_lower.find(entity_lower, start_pos)
        if actual_start == -1:
            actual_start = start_pos
        
        actual_end = actual_start + len(entity_text)
        
        return {
            "label": entity_type,
            "text": entity_text,
            "start": actual_start,
            "end": actual_end
        }
    
    def generate_iob2_labels(text, entities):
        """Tạo IOB2 labels từ entities"""
        words = text.split()
        labels = ["O"] * len(words)
        
        # Sort entities by start position
        entities.sort(key=lambda x: x['start'])
        
        current_char_idx = 0
        for i, word in enumerate(words):
            word_len = len(word)
            word_start = current_char_idx
            word_end = current_char_idx + word_len
            
            # Check if any entity starts within this word
            for entity in entities:
                entity_start = entity['start']
                entity_end = entity['end']
                
                if entity_start >= word_start and entity_start < word_end:
                    # Entity starts in this word
                    labels[i] = f"B-{entity['label']}"
                    break
                elif entity_start < word_start and entity_end > word_start:
                    # Entity continues in this word
                    labels[i] = f"I-{entity['label']}"
                    break
            
            current_char_idx += word_len + 1  # +1 for space
        
        return labels
    
    # Process dataset
    print("🔄 Processing dataset...")
    updated_dataset = []
    entity_stats = defaultdict(int)
    
    for i, sample in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(dataset)} samples...")
        
        text = sample.get('input', '')
        command = sample.get('command', 'unknown')
        
        # Get existing entities
        existing_entities = {}
        for entity in sample.get('entities', []):
            entity_type = entity.get('label', '')
            if entity_type:
                existing_entities[entity_type] = entity.get('text', '')
        
        # Extract missing entities
        enhanced_entities = extract_missing_entities(text, command, existing_entities)
        
        # Create new entities list
        new_entities = []
        for entity_type, entity_text in enhanced_entities.items():
            if entity_text:
                # Tìm vị trí của entity trong text
                entity_lower = entity_text.lower()
                text_lower = text.lower()
                start_pos = text_lower.find(entity_lower)
                
                if start_pos != -1:
                    entity_obj = create_entity_from_text(text, entity_text, entity_type, start_pos)
                    new_entities.append(entity_obj)
                    entity_stats[entity_type] += 1
        
        # Generate IOB2 labels
        iob2_labels = generate_iob2_labels(text, new_entities)
        
        # Create updated sample
        updated_sample = sample.copy()
        updated_sample['entities'] = new_entities
        updated_sample['bio_labels'] = iob2_labels
        
        # Remove spans if not needed
        if 'spans' in updated_sample:
            del updated_sample['spans']
        
        updated_dataset.append(updated_sample)
    
    print(f"✅ Processed {len(updated_dataset)} samples")
    
    # Show entity statistics
    print("\n📊 ENTITY STATISTICS:")
    for entity_type, count in sorted(entity_stats.items()):
        print(f"  {entity_type}: {count}")
    
    # Save updated dataset
    print("\n💾 Saving updated dataset...")
    with open(main_dataset_path, 'w', encoding='utf-8') as f:
        json.dump(updated_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved updated dataset to {main_dataset_path}")
    
    # Delete redundant files
    print("\n🗑️ Cleaning up redundant files...")
    redundant_files = [
        "src/data/raw/elderly_command_dataset_MERGED_13C_VITEXT_UPDATED_IOB2.json"
    ]
    
    for file_path in redundant_files:
        if Path(file_path).exists():
            Path(file_path).unlink()
            print(f"✅ Deleted {file_path}")
    
    print("\n🎉 MAIN DATASET UPDATED SUCCESSFULLY!")
    print("=" * 60)
    print("Dataset is now ready for training with correct command-entity mapping and IOB2 labels!")

if __name__ == "__main__":
    update_main_dataset()
