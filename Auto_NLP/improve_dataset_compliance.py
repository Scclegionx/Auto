#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cải thiện dataset compliance một cách cẩn thận và hiệu quả
"""

import json
import re
from collections import defaultdict, Counter

def improve_dataset_compliance():
    """Cải thiện dataset compliance"""
    
    print("🔧 IMPROVING DATASET COMPLIANCE")
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
    
    # Load dataset
    dataset_path = "src/data/raw/elderly_command_dataset_MERGED_13C_VITEXT.json"
    print(f"📖 Loading dataset from {dataset_path}...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Enhanced extraction patterns
    EXTRACTION_PATTERNS = {
        "PLATFORM": [
            # Explicit platform mentions
            r"(zalo|viber|skype|facetime|messenger|telegram|whatsapp|google|bing|youtube|sms|tin nhắn|text)",
            r"(qua|trên|bằng|dùng|sử dụng)\s+(zalo|viber|skype|facetime|messenger|telegram|whatsapp|google|bing|youtube|sms|tin nhắn|text)",
            # Context-based platform detection
            r"(gọi|nhắn|gửi|tìm|search)\s+(zalo|viber|skype|facetime|messenger|telegram|whatsapp|google|bing|youtube|sms|tin nhắn|text)",
            # Default platforms for specific commands
            r"(gọi|call|phone)"  # Default to "phone" for call commands
        ],
        "DATE": [
            # Time expressions
            r"(hôm nay|mai|ngày mai|hôm qua|sáng|chiều|tối|đêm)",
            r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})",
            r"(\d{1,2})\s+(tháng|month)\s+(\d{4})",
            r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})",
            r"(ngày|tháng|năm)\s+(\d+)",
            # Relative dates
            r"(lúc|vào|khi)\s+(\d{1,2})\s*(giờ|h)",
            # Default to "today" for alarm commands
            r"(báo thức|alarm|đặt|set)"
        ],
        "ACTION": [
            # Camera actions
            r"(mở|bật|tắt|đóng|chụp|quay)\s+(camera|cam|máy ảnh)",
            r"(mở|bật|tắt|đóng)\s+(camera|cam)",
            # General actions
            r"(bật|tắt|mở|đóng|tăng|giảm|on|off|up|down)",
            # Default action for open-cam
            r"(camera|cam|máy ảnh)"
        ]
    }
    
    def extract_missing_entities(text, command, existing_entities):
        """Extract missing entities với patterns cải tiến"""
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
                                # Clean up entity text
                                if entity_type == "PLATFORM":
                                    # Extract just the platform name
                                    platform_match = re.search(r"(zalo|viber|skype|facetime|messenger|telegram|whatsapp|google|bing|youtube|sms|tin nhắn|text|phone)", entity_text, re.IGNORECASE)
                                    if platform_match:
                                        entity_text = platform_match.group(1)
                                    elif "gọi" in entity_text.lower() or "call" in entity_text.lower():
                                        entity_text = "phone"
                                elif entity_type == "DATE":
                                    # Extract date information
                                    if "hôm nay" in entity_text.lower():
                                        entity_text = "hôm nay"
                                    elif "mai" in entity_text.lower():
                                        entity_text = "ngày mai"
                                elif entity_type == "ACTION":
                                    # Extract action
                                    if "mở" in entity_text.lower() or "bật" in entity_text.lower():
                                        entity_text = "mở"
                                    elif "tắt" in entity_text.lower() or "đóng" in entity_text.lower():
                                        entity_text = "tắt"
                                    elif "camera" in entity_text.lower() or "cam" in entity_text.lower():
                                        entity_text = "mở"
                                
                                enhanced_entities[entity_type] = entity_text
                                break
                        if entity_type in enhanced_entities and enhanced_entities[entity_type]:
                            break
        
        return enhanced_entities
    
    def create_entity_from_text(text, entity_text, entity_type, start_pos):
        """Tạo entity object từ text"""
        entity_lower = entity_text.lower()
        text_lower = text.lower()
        
        # Tìm vị trí thực tế của entity trong text
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
    
    def filter_entities_by_command(entities, command):
        """Filter entities chỉ giữ lại những cái cần thiết cho command"""
        required_entities = set(REQUIRED_MAPPING.get(command, []))
        filtered_entities = []
        
        for entity in entities:
            if entity['label'] in required_entities:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    # Process dataset
    print("🔄 Processing dataset...")
    updated_dataset = []
    entity_stats = defaultdict(int)
    improvement_stats = defaultdict(int)
    
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
                    
                    # Track improvements
                    if entity_type not in existing_entities:
                        improvement_stats[entity_type] += 1
        
        # Filter entities to only include required ones
        filtered_entities = filter_entities_by_command(new_entities, command)
        
        # Generate IOB2 labels
        iob2_labels = generate_iob2_labels(text, filtered_entities)
        
        # Create updated sample
        updated_sample = sample.copy()
        updated_sample['entities'] = filtered_entities
        updated_sample['bio_labels'] = iob2_labels
        
        # Remove spans if not needed
        if 'spans' in updated_sample:
            del updated_sample['spans']
        
        updated_dataset.append(updated_sample)
    
    print(f"✅ Processed {len(updated_dataset)} samples")
    
    # Show improvement statistics
    print("\n📊 IMPROVEMENT STATISTICS:")
    print("-" * 40)
    for entity_type, count in sorted(improvement_stats.items()):
        print(f"  {entity_type}: +{count} new entities")
    
    print("\n📊 FINAL ENTITY STATISTICS:")
    print("-" * 40)
    for entity_type, count in sorted(entity_stats.items()):
        print(f"  {entity_type}: {count}")
    
    # Save updated dataset
    print("\n💾 Saving improved dataset...")
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(updated_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved improved dataset to {dataset_path}")
    
    print("\n🎉 DATASET IMPROVEMENT COMPLETED!")
    print("=" * 60)
    print("Dataset has been improved with better entity extraction and filtering!")

if __name__ == "__main__":
    improve_dataset_compliance()
