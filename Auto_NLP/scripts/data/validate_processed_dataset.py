#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate processed dataset format để đảm bảo tương thích với training pipeline
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter

# Add project root to path
# File location: <project_root>/scripts/data/validate_processed_dataset.py
# parents[0] = scripts/data
# parents[1] = scripts
# parents[2] = project_root
_script_file = Path(__file__).resolve()
PROJECT_ROOT = _script_file.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

# Debug: print paths
# print(f"Script: {_script_file}")
# print(f"Project root: {PROJECT_ROOT}")
# print(f"Entity schema: {PROJECT_ROOT / 'src' / 'data' / 'entity_schema.py'}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import with absolute path
import importlib.util
entity_schema_path = PROJECT_ROOT / "src" / "data" / "entity_schema.py"
if not entity_schema_path.exists():
    raise FileNotFoundError(f"Entity schema not found at: {entity_schema_path}")

spec = importlib.util.spec_from_file_location("entity_schema", entity_schema_path)
entity_schema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entity_schema)
ENTITY_BASE_NAMES = entity_schema.ENTITY_BASE_NAMES

config_path = PROJECT_ROOT / "src" / "training" / "configs" / "config.py"
if not config_path.exists():
    raise FileNotFoundError(f"Config not found at: {config_path}")

spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
IntentConfig = config_module.IntentConfig
CommandConfig = config_module.CommandConfig

# Expected fields
REQUIRED_FIELDS = {"input", "command", "intent", "entities", "spans"}
OPTIONAL_FIELDS = {"bio_labels", "meta", "split", "tokens"}

# Valid entity labels (BIO format)
VALID_ENTITY_LABELS = set(["O"] + [f"B-{name}" for name in ENTITY_BASE_NAMES] + [f"I-{name}" for name in ENTITY_BASE_NAMES])

# Valid intents/commands
intent_config = IntentConfig()
command_config = CommandConfig()
VALID_INTENTS = set(intent_config.intent_labels)
VALID_COMMANDS = set(command_config.command_labels)


def validate_sample(sample: Dict, idx: int, file_path: Path) -> List[str]:
    """Validate một sample và trả về list errors"""
    errors = []
    
    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in sample:
            errors.append(f"Sample {idx}: Missing required field '{field}'")
    
    if errors:
        return errors  # Skip further checks if missing required fields
    
    # Check input field
    if not isinstance(sample["input"], str) or not sample["input"].strip():
        errors.append(f"Sample {idx}: 'input' must be non-empty string")
    
    text = sample["input"]
    text_len = len(text)
    
    # Check command/intent
    if sample["command"] not in VALID_COMMANDS:
        errors.append(f"Sample {idx}: Invalid command '{sample['command']}' (not in {VALID_COMMANDS})")
    
    if sample["intent"] not in VALID_INTENTS:
        errors.append(f"Sample {idx}: Invalid intent '{sample['intent']}' (not in {VALID_INTENTS})")
    
    # Check entities
    if not isinstance(sample["entities"], list):
        errors.append(f"Sample {idx}: 'entities' must be a list")
    else:
        for ent_idx, entity in enumerate(sample["entities"]):
            if not isinstance(entity, dict):
                errors.append(f"Sample {idx}: entities[{ent_idx}] must be a dict")
                continue
            
            # Check entity fields
            if "label" not in entity:
                errors.append(f"Sample {idx}: entities[{ent_idx}] missing 'label'")
            elif entity["label"] not in ENTITY_BASE_NAMES:
                errors.append(f"Sample {idx}: entities[{ent_idx}] invalid label '{entity['label']}'")
            
            if "text" not in entity:
                errors.append(f"Sample {idx}: entities[{ent_idx}] missing 'text'")
            
            if "start" not in entity or "end" not in entity:
                errors.append(f"Sample {idx}: entities[{ent_idx}] missing 'start' or 'end'")
            else:
                start = entity.get("start", -1)
                end = entity.get("end", -1)
                if not isinstance(start, int) or not isinstance(end, int):
                    errors.append(f"Sample {idx}: entities[{ent_idx}] 'start'/'end' must be int")
                elif start < 0 or end < 0:
                    errors.append(f"Sample {idx}: entities[{ent_idx}] 'start'/'end' must be >= 0")
                elif start >= end:
                    errors.append(f"Sample {idx}: entities[{ent_idx}] 'start' ({start}) >= 'end' ({end})")
                elif end > text_len:
                    errors.append(f"Sample {idx}: entities[{ent_idx}] 'end' ({end}) > text length ({text_len})")
    
    # Check spans (similar to entities)
    if not isinstance(sample["spans"], list):
        errors.append(f"Sample {idx}: 'spans' must be a list")
    else:
        for span_idx, span in enumerate(sample["spans"]):
            if not isinstance(span, dict):
                errors.append(f"Sample {idx}: spans[{span_idx}] must be a dict")
                continue
            
            # Check span fields
            if "label" not in span:
                errors.append(f"Sample {idx}: spans[{span_idx}] missing 'label'")
            elif span["label"] not in ENTITY_BASE_NAMES:
                errors.append(f"Sample {idx}: spans[{span_idx}] invalid label '{span['label']}'")
            
            if "text" not in span:
                errors.append(f"Sample {idx}: spans[{span_idx}] missing 'text'")
            
            if "start" not in span or "end" not in span:
                errors.append(f"Sample {idx}: spans[{span_idx}] missing 'start' or 'end'")
            else:
                start = span.get("start", -1)
                end = span.get("end", -1)
                if not isinstance(start, int) or not isinstance(end, int):
                    errors.append(f"Sample {idx}: spans[{span_idx}] 'start'/'end' must be int")
                elif start < 0 or end < 0:
                    errors.append(f"Sample {idx}: spans[{span_idx}] 'start'/'end' must be >= 0")
                elif start >= end:
                    errors.append(f"Sample {idx}: spans[{span_idx}] 'start' ({start}) >= 'end' ({end})")
                elif end > text_len:
                    errors.append(f"Sample {idx}: spans[{span_idx}] 'end' ({end}) > text length ({text_len})")
    
    return errors


def validate_dataset(file_path: Path) -> Dict:
    """Validate toàn bộ dataset file"""
    print(f"\nValidating {file_path.name}...")
    
    # Load dataset
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {
            "valid": False,
            "error": f"Failed to load file: {e}",
            "stats": {}
        }
    
    if not isinstance(data, list):
        return {
            "valid": False,
            "error": "Dataset must be a list of samples",
            "stats": {}
        }
    
    # Validate each sample
    all_errors = []
    stats = {
        "total_samples": len(data),
        "valid_samples": 0,
        "invalid_samples": 0,
        "command_distribution": Counter(),
        "intent_distribution": Counter(),
        "entity_type_distribution": Counter(),
        "avg_text_length": 0,
        "avg_entities_per_sample": 0,
        "avg_spans_per_sample": 0,
    }
    
    total_text_length = 0
    total_entities = 0
    total_spans = 0
    
    for idx, sample in enumerate(data):
        errors = validate_sample(sample, idx, file_path)
        if errors:
            all_errors.extend(errors)
            stats["invalid_samples"] += 1
        else:
            stats["valid_samples"] += 1
            stats["command_distribution"][sample.get("command", "unknown")] += 1
            stats["intent_distribution"][sample.get("intent", "unknown")] += 1
            
            # Count entities
            entities = sample.get("entities", [])
            total_entities += len(entities)
            for entity in entities:
                if isinstance(entity, dict):
                    label = entity.get("label", "UNKNOWN")
                    stats["entity_type_distribution"][label] += 1
            
            # Count spans
            spans = sample.get("spans", [])
            total_spans += len(spans)
            
            # Text length
            text = sample.get("input", "")
            total_text_length += len(text)
    
    # Calculate averages
    if stats["valid_samples"] > 0:
        stats["avg_text_length"] = total_text_length / stats["valid_samples"]
        stats["avg_entities_per_sample"] = total_entities / stats["valid_samples"]
        stats["avg_spans_per_sample"] = total_spans / stats["valid_samples"]
    
    return {
        "valid": len(all_errors) == 0,
        "errors": all_errors[:50],  # Limit to first 50 errors
        "total_errors": len(all_errors),
        "stats": stats
    }


def print_stats(stats: Dict, file_name: str):
    """Print statistics"""
    print(f"\nStatistics for {file_name}:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Valid samples: {stats['valid_samples']}")
    print(f"  Invalid samples: {stats['invalid_samples']}")
    print(f"  Average text length: {stats['avg_text_length']:.1f} chars")
    print(f"  Average entities per sample: {stats['avg_entities_per_sample']:.2f}")
    print(f"  Average spans per sample: {stats['avg_spans_per_sample']:.2f}")
    
    print(f"\n  Command distribution:")
    for cmd, count in stats['command_distribution'].most_common():
        print(f"    {cmd}: {count}")
    
    print(f"\n  Top 10 entity types:")
    for entity_type, count in stats['entity_type_distribution'].most_common(10):
        print(f"    {entity_type}: {count}")


def main():
    """Main validation function"""
    # Set UTF-8 encoding for Windows
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("=" * 70)
    print("PROCESSED DATASET VALIDATION")
    print("=" * 70)
    
    # Dataset paths
    processed_dir = PROJECT_ROOT / "src" / "data" / "processed"
    train_path = processed_dir / "train.json"
    val_path = processed_dir / "val.json"
    test_path = processed_dir / "test.json"
    
    # Validate each file
    results = {}
    for file_path in [train_path, val_path, test_path]:
        if not file_path.exists():
            print(f"\n[ERROR] File not found: {file_path}")
            results[file_path.name] = {"valid": False, "error": "File not found"}
            continue
        
        result = validate_dataset(file_path)
        results[file_path.name] = result
        
        if result["valid"]:
            print(f"[OK] {file_path.name}: VALID ({result['stats']['valid_samples']} samples)")
            print_stats(result["stats"], file_path.name)
        else:
            print(f"[ERROR] {file_path.name}: INVALID")
            print(f"   Total errors: {result['total_errors']}")
            if result.get("error"):
                print(f"   Error: {result['error']}")
            if result.get("errors"):
                print(f"\n   First 10 errors:")
                for error in result["errors"][:10]:
                    print(f"     - {error}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    all_valid = all(r.get("valid", False) for r in results.values())
    
    for file_name, result in results.items():
        status = "[OK] VALID" if result.get("valid", False) else "[ERROR] INVALID"
        print(f"  {file_name}: {status}")
        if not result.get("valid", False):
            if result.get("error"):
                print(f"    Error: {result['error']}")
            if result.get("total_errors", 0) > 0:
                print(f"    Total errors: {result['total_errors']}")
    
    if all_valid:
        print("\n[SUCCESS] All datasets are valid! Ready for training.")
        return 0
    else:
        print("\n[WARNING] Some datasets have errors. Please fix them before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

