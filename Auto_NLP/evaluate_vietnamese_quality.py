#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Đánh giá chất lượng tiếng Việt của dataset
"""

import json
import re
from collections import Counter, defaultdict
import random

def evaluate_vietnamese_quality():
    """Đánh giá chất lượng tiếng Việt của dataset"""
    
    print("🇻🇳 EVALUATING VIETNAMESE LANGUAGE QUALITY")
    print("=" * 60)
    
    # Load dataset
    dataset_path = "src/data/raw/elderly_command_dataset_MERGED_13C_VITEXT.json"
    print(f"📖 Loading dataset from {dataset_path}...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Sample analysis
    sample_size = min(1000, len(dataset))
    sample_data = random.sample(dataset, sample_size)
    
    print(f"\n📊 ANALYZING {sample_size} RANDOM SAMPLES")
    print("-" * 50)
    
    # 1. Text length analysis
    text_lengths = [len(item['input']) for item in sample_data]
    avg_length = sum(text_lengths) / len(text_lengths)
    min_length = min(text_lengths)
    max_length = max(text_lengths)
    
    print(f"📏 TEXT LENGTH ANALYSIS:")
    print(f"   Average length: {avg_length:.1f} characters")
    print(f"   Min length: {min_length}")
    print(f"   Max length: {max_length}")
    
    # 2. Word count analysis
    word_counts = [len(item['input'].split()) for item in sample_data]
    avg_words = sum(word_counts) / len(word_counts)
    
    print(f"\n📝 WORD COUNT ANALYSIS:")
    print(f"   Average words: {avg_words:.1f} words per sentence")
    
    # 3. Language patterns analysis
    print(f"\n🔍 LANGUAGE PATTERNS ANALYSIS:")
    
    # Common Vietnamese patterns
    vietnamese_patterns = {
        "Politeness": [
            r"(ạ|ạ ạ|ạ ạ ạ|ạ ạ ạ ạ)",
            r"(dạ|dạ ạ|dạ ạ ạ)",
            r"(vâng|vâng ạ|vâng ạ ạ)",
            r"(cảm ơn|cảm ơn ạ|cảm ơn ạ ạ)",
            r"(xin lỗi|xin lỗi ạ|xin lỗi ạ ạ)"
        ],
        "Elderly_speech": [
            r"(ờ|ờ ờ|ờ ờ ờ)",
            r"(nè|nè nè|nè nè nè)",
            r"(ê|ê ê|ê ê ê)",
            r"(anh ơi|chị ơi|em ơi|cô ơi|chú ơi)",
            r"(mình|mình ơi|mình à)"
        ],
        "Hesitation": [
            r"(à|à à|à à à)",
            r"(ừm|ừm ừm|ừm ừm ừm)",
            r"(hmm|hmm hmm|hmm hmm hmm)",
            r"(thì|thì thì|thì thì thì)"
        ],
        "Repetition": [
            r"(\b\w+)\s+\1\b",  # Word repetition
            r"(\b\w+\s+\w+)\s+\1\b"  # Phrase repetition
        ]
    }
    
    pattern_counts = defaultdict(int)
    total_samples = len(sample_data)
    
    for pattern_type, patterns in vietnamese_patterns.items():
        count = 0
        for item in sample_data:
            text = item['input']
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    count += 1
                    break
        pattern_counts[pattern_type] = count
        percentage = (count / total_samples) * 100
        print(f"   {pattern_type}: {count}/{total_samples} ({percentage:.1f}%)")
    
    # 4. Command-specific language analysis
    print(f"\n🎯 COMMAND-SPECIFIC LANGUAGE ANALYSIS:")
    
    command_samples = defaultdict(list)
    for item in sample_data:
        command = item['command']
        command_samples[command].append(item['input'])
    
    for command, samples in command_samples.items():
        if len(samples) >= 5:  # Only analyze commands with enough samples
            print(f"\n   📋 {command} ({len(samples)} samples):")
            
            # Show sample texts
            for i, sample in enumerate(samples[:3]):  # Show first 3 samples
                print(f"      {i+1}. {sample}")
            
            # Analyze language characteristics
            avg_len = sum(len(s) for s in samples) / len(samples)
            print(f"      Average length: {avg_len:.1f} chars")
    
    # 5. Quality indicators
    print(f"\n⭐ QUALITY INDICATORS:")
    
    # Check for common issues
    issues = {
        "Too_short": sum(1 for item in sample_data if len(item['input']) < 10),
        "Too_long": sum(1 for item in sample_data if len(item['input']) > 200),
        "No_vietnamese": sum(1 for item in sample_data if not re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', item['input'], re.IGNORECASE)),
        "Repetitive": sum(1 for item in sample_data if len(set(item['input'].split())) < len(item['input'].split()) * 0.5),
        "Incomplete": sum(1 for item in sample_data if item['input'].endswith(('...', '..', '.', '?', '!')) == False)
    }
    
    for issue, count in issues.items():
        percentage = (count / total_samples) * 100
        status = "❌" if percentage > 20 else "⚠️" if percentage > 10 else "✅"
        print(f"   {status} {issue}: {count}/{total_samples} ({percentage:.1f}%)")
    
    # 6. Elderly-friendly features
    print(f"\n👴 ELDERLY-FRIENDLY FEATURES:")
    
    elderly_features = {
        "Simple_vocabulary": sum(1 for item in sample_data if len([w for w in item['input'].split() if len(w) > 8]) < 3),
        "Common_words": sum(1 for item in sample_data if any(word in item['input'].lower() for word in ['gọi', 'nhắn', 'phát', 'tìm', 'đặt', 'mở', 'bật', 'tắt'])),
        "Polite_tone": sum(1 for item in sample_data if any(word in item['input'].lower() for word in ['ạ', 'dạ', 'vâng', 'cảm ơn', 'xin lỗi'])),
        "Clear_commands": sum(1 for item in sample_data if any(word in item['input'].lower() for word in ['cho', 'tôi', 'giúp', 'làm', 'thực hiện']))
    }
    
    for feature, count in elderly_features.items():
        percentage = (count / total_samples) * 100
        status = "✅" if percentage > 70 else "⚠️" if percentage > 50 else "❌"
        print(f"   {status} {feature}: {count}/{total_samples} ({percentage:.1f}%)")
    
    # 7. Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print("-" * 50)
    
    # Calculate quality score
    quality_score = 0
    
    # Length appropriateness (20 points)
    if 20 <= avg_length <= 100:
        quality_score += 20
    elif 15 <= avg_length <= 150:
        quality_score += 15
    else:
        quality_score += 10
    
    # Vietnamese content (20 points)
    vietnamese_ratio = (total_samples - issues["No_vietnamese"]) / total_samples
    quality_score += int(vietnamese_ratio * 20)
    
    # Elderly-friendly features (30 points)
    elderly_score = sum(elderly_features.values()) / len(elderly_features) / total_samples * 100
    quality_score += int(elderly_score * 0.3)
    
    # Pattern richness (15 points)
    pattern_richness = sum(pattern_counts.values()) / len(pattern_counts) / total_samples * 100
    quality_score += int(pattern_richness * 0.15)
    
    # Issue penalty (15 points)
    issue_penalty = sum(issues.values()) / len(issues) / total_samples * 100
    quality_score += max(0, 15 - int(issue_penalty * 0.15))
    
    quality_score = min(100, quality_score)
    
    print(f"📊 Quality Score: {quality_score}/100")
    
    if quality_score >= 80:
        print("🎉 EXCELLENT! Dataset is very suitable for elderly users!")
    elif quality_score >= 70:
        print("✅ GOOD! Dataset is suitable for elderly users with minor improvements needed.")
    elif quality_score >= 60:
        print("⚠️ FAIR! Dataset needs some improvements for elderly users.")
    else:
        print("❌ POOR! Dataset needs significant improvements for elderly users.")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if issues["Too_short"] > total_samples * 0.1:
        print("   - Add more descriptive inputs for short commands")
    if issues["No_vietnamese"] > total_samples * 0.05:
        print("   - Ensure all inputs contain Vietnamese characters")
    if elderly_score < 70:
        print("   - Add more elderly-friendly expressions and politeness markers")
    if pattern_richness < 50:
        print("   - Increase variety in speech patterns and expressions")
    
    return quality_score

if __name__ == "__main__":
    evaluate_vietnamese_quality()
