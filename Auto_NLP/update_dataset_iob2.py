#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cập nhật dataset với IOB2 labels đúng ngữ nghĩa theo yêu cầu
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import re

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class DatasetIOB2Updater:
    """Cập nhật dataset với IOB2 labels mới"""
    
    def __init__(self):
        # Entity mapping mới theo yêu cầu
        self.command_entity_mapping = {
            "add-contacts": {
                "required": ["CONTACT_NAME", "PHONE"],
                "optional": [],
                "description": "Thêm liên hệ mới"
            },
            "call": {
                "required": ["CONTACT_NAME"],
                "optional": ["PHONE", "RECEIVER", "PLATFORM"],
                "description": "Gọi điện thoại"
            },
            "make-video-call": {
                "required": ["CONTACT_NAME"],
                "optional": ["RECEIVER", "PLATFORM"],
                "description": "Gọi video"
            },
            "send-mess": {
                "required": ["RECEIVER", "MESSAGE"],
                "optional": ["PLATFORM"],
                "description": "Gửi tin nhắn"
            },
            "set-alarm": {
                "required": ["TIME"],
                "optional": ["DATE"],
                "description": "Đặt báo thức"
            },
            "set-event-calendar": {
                "required": ["TITLE", "DATE"],
                "optional": ["TIME"],
                "description": "Đặt lịch sự kiện"
            },
            "play-media": {
                "required": ["MEDIA_TYPE"],
                "optional": ["CONTENT_TYPE"],
                "description": "Phát media"
            },
            "view-content": {
                "required": ["CONTENT_TYPE"],
                "optional": ["PLATFORM"],
                "description": "Xem nội dung"
            },
            "search-internet": {
                "required": ["QUERY"],
                "optional": ["PLATFORM"],
                "description": "Tìm kiếm internet"
            },
            "search-youtube": {
                "required": ["QUERY"],
                "optional": ["PLATFORM"],
                "description": "Tìm kiếm YouTube"
            },
            "get-info": {
                "required": ["QUERY"],
                "optional": ["CONTENT_TYPE", "PLATFORM"],
                "description": "Lấy thông tin"
            },
            "control-device": {
                "required": ["ACTION", "DEVICE"],
                "optional": ["MODE", "LOCATION"],
                "description": "Điều khiển thiết bị"
            },
            "open-cam": {
                "required": ["ACTION"],
                "optional": ["MODE", "CAMERA_TYPE"],
                "description": "Mở camera"
            }
        }
        
        # Entity labels mới (BIO2 format)
        self.new_entity_labels = [
            "O",  # Outside
            # B- labels (Begin)
            "B-ACTION", "B-CAMERA_TYPE", "B-CONTACT_NAME", "B-CONTENT_TYPE", 
            "B-DATE", "B-DEVICE", "B-LOCATION", "B-MEDIA_TYPE", "B-MESSAGE", 
            "B-MODE", "B-PHONE", "B-PLATFORM", "B-QUERY", "B-RECEIVER", 
            "B-TIME", "B-TITLE",
            # I- labels (Inside)
            "I-ACTION", "I-CAMERA_TYPE", "I-CONTACT_NAME", "I-CONTENT_TYPE", 
            "I-DATE", "I-DEVICE", "I-LOCATION", "I-MEDIA_TYPE", "I-MESSAGE", 
            "I-MODE", "I-PHONE", "I-PLATFORM", "I-QUERY", "I-RECEIVER", 
            "I-TIME", "I-TITLE"
        ]
        
        # Mapping từ entity cũ sang mới
        self.entity_mapping = {
            # Giữ nguyên
            "CONTACT_NAME": "CONTACT_NAME",
            "PHONE": "PHONE",
            "RECEIVER": "RECEIVER",
            "MESSAGE": "MESSAGE",
            "PLATFORM": "PLATFORM",
            "TIME": "TIME",
            "DATE": "DATE",
            "TITLE": "TITLE",
            "ACTION": "ACTION",
            "DEVICE": "DEVICE",
            "MODE": "MODE",
            "CAMERA_TYPE": "CAMERA_TYPE",
            "MEDIA_TYPE": "MEDIA_TYPE",
            "CONTENT_TYPE": "CONTENT_TYPE",
            "QUERY": "QUERY",
            "LOCATION": "LOCATION",
            # Loại bỏ
            "LEVEL": None,  # Không cần thiết
        }
    
    def load_dataset(self, file_path):
        """Load dataset"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_sample_entities(self, sample):
        """Cập nhật entities cho một sample"""
        command = sample.get('command', sample.get('intent', ''))
        text = sample['input']
        
        if command not in self.command_entity_mapping:
            return sample
        
        command_config = self.command_entity_mapping[command]
        allowed_entities = set(command_config['required'] + command_config['optional'])
        
        # Filter spans chỉ giữ lại entities được phép
        new_spans = []
        for span in sample.get('spans', []):
            entity_label = span['label']
            
            # Map entity label
            mapped_label = self.entity_mapping.get(entity_label, entity_label)
            
            # Chỉ giữ lại nếu entity được phép cho command này
            if mapped_label and mapped_label in allowed_entities:
                new_span = span.copy()
                new_span['label'] = mapped_label
                new_spans.append(new_span)
        
        # Update sample
        updated_sample = sample.copy()
        updated_sample['spans'] = new_spans
        
        # Update entities list
        new_entities = []
        for entity in sample.get('entities', []):
            entity_label = entity['label']
            mapped_label = self.entity_mapping.get(entity_label, entity_label)
            
            if mapped_label and mapped_label in allowed_entities:
                new_entity = entity.copy()
                new_entity['label'] = mapped_label
                new_entities.append(new_entity)
        
        updated_sample['entities'] = new_entities
        
        # Regenerate BIO2 labels
        updated_sample['bio_labels'] = self.generate_bio2_labels(text, new_spans)
        
        return updated_sample
    
    def generate_bio2_labels(self, text, spans):
        """Generate BIO2 labels từ spans"""
        # Tokenize text thành words
        words = text.split()
        bio_labels = ['O'] * len(words)
        
        # Sort spans by start position
        sorted_spans = sorted(spans, key=lambda x: x['start'])
        
        for span in sorted_spans:
            start = span['start']
            end = span['end']
            label = span['label']
            span_text = text[start:end]
            
            # Find word positions for this span
            word_start = None
            word_end = None
            
            char_pos = 0
            for i, word in enumerate(words):
                word_start_pos = char_pos
                word_end_pos = char_pos + len(word)
                
                # Check if span overlaps with this word
                if (start < word_end_pos and end > word_start_pos):
                    if word_start is None:
                        word_start = i
                    word_end = i
                
                char_pos = word_end_pos + 1  # +1 for space
            
            # Assign BIO2 labels
            if word_start is not None and word_end is not None:
                for i in range(word_start, word_end + 1):
                    if i < len(bio_labels):
                        if i == word_start:
                            bio_labels[i] = f"B-{label}"
                        else:
                            bio_labels[i] = f"I-{label}"
        
        return bio_labels
    
    def update_dataset(self, data):
        """Cập nhật toàn bộ dataset"""
        updated_data = []
        
        print(f"Updating {len(data)} samples...")
        
        for i, sample in enumerate(data):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(data)} samples...")
            
            updated_sample = self.update_sample_entities(sample)
            updated_data.append(updated_sample)
        
        print(f"✅ Updated {len(updated_data)} samples")
        return updated_data
    
    def analyze_updated_dataset(self, data):
        """Phân tích dataset sau khi cập nhật"""
        print(f"\n📊 UPDATED DATASET ANALYSIS")
        print("=" * 60)
        
        # Entity distribution
        all_entities = []
        for sample in data:
            for span in sample.get('spans', []):
                all_entities.append(span['label'])
        
        entity_counts = {}
        for entity in all_entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        print(f"Entity Distribution:")
        for entity, count in sorted(entity_counts.items()):
            print(f"   {entity}: {count}")
        
        # Command-entity mapping
        command_entities = defaultdict(set)
        for sample in data:
            command = sample.get('command', 'unknown')
            for span in sample.get('spans', []):
                command_entities[command].add(span['label'])
        
        print(f"\nCommand-Entity Mapping:")
        for command, entities in command_entities.items():
            print(f"   {command}: {sorted(entities)}")
        
        # BIO2 label distribution
        all_bio_labels = []
        for sample in data:
            all_bio_labels.extend(sample.get('bio_labels', []))
        
        bio_counts = {}
        for label in all_bio_labels:
            bio_counts[label] = bio_counts.get(label, 0) + 1
        
        print(f"\nBIO2 Label Distribution:")
        for label, count in sorted(bio_counts.items()):
            print(f"   {label}: {count}")

def main():
    """Main function"""
    print("🔄 UPDATING DATASET WITH IOB2 LABELS")
    print("=" * 60)
    
    # Paths
    input_file = "src/data/raw/elderly_command_dataset_MERGED_13C_VITEXT.json"
    output_file = "src/data/raw/elderly_command_dataset_MERGED_13C_VITEXT_UPDATED_IOB2.json"
    
    try:
        # Create updater
        updater = DatasetIOB2Updater()
        
        # Load dataset
        print("📖 Loading dataset...")
        data = updater.load_dataset(input_file)
        print(f"✅ Loaded {len(data)} samples")
        
        # Update dataset
        print("\n🔄 Updating dataset...")
        updated_data = updater.update_dataset(data)
        
        # Analyze updated dataset
        updater.analyze_updated_dataset(updated_data)
        
        # Save updated dataset
        print(f"\n💾 Saving updated dataset...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved to {output_file}")
        
        print(f"\n🎉 DATASET UPDATE COMPLETED!")
        print("=" * 60)
        print("Dataset is ready for training with new IOB2 labels!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
