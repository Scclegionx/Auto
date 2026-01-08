#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Tool cho Intent/Command Classification và NER Results
Tạo các biểu đồ và báo cáo trực quan để đánh giá model performance
"""

import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, f1_score
)
from seqeval.metrics import (
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
    classification_report as seqeval_classification_report
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_loader import TrainedModelInference
from src.data.processed.data_processor import DataProcessor
from src.training.configs.config import IntentConfig, CommandConfig, EntityConfig, ModelConfig

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set font for Vietnamese
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class ModelResultsVisualizer:
    """Visualizer cho Intent/Command Classification và NER Results"""
    
    def __init__(self, model_path: str = "models/phobert_multitask", test_data_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path) if test_data_path else Path(ModelConfig().test_data_path)
        
        # Load model
        print(f"📦 Loading model from: {self.model_path}")
        self.model_inference = TrainedModelInference(str(self.model_path))
        
        # Load configs
        self.intent_config = IntentConfig()
        self.command_config = CommandConfig()
        self.entity_config = EntityConfig()
        
        # Create label2id mappings
        self.intent_label2id = {label: idx for idx, label in enumerate(self.intent_config.intent_labels)}
        self.intent_id2label = {idx: label for label, idx in self.intent_label2id.items()}
        
        self.command_label2id = {label: idx for idx, label in enumerate(self.command_config.command_labels)}
        self.command_id2label = {idx: label for label, idx in self.command_label2id.items()}
        
        self.entity_label2id = {label: idx for idx, label in enumerate(self.entity_config.entity_labels)}
        self.entity_id2label = {idx: label for label, idx in self.entity_label2id.items()}
        
        # Load test data
        print(f"📚 Loading test data from: {self.test_data_path}")
        self.processor = DataProcessor()
        self.test_data = self.processor.load_dataset(str(self.test_data_path))
        print(f"✅ Loaded {len(self.test_data)} test samples")
        
        # Storage for predictions
        self.intent_predictions: List[int] = []
        self.intent_labels: List[int] = []
        self.command_predictions: List[int] = []
        self.command_labels: List[int] = []
        self.entity_predictions: List[List[str]] = []
        self.entity_labels: List[List[str]] = []
        self.texts: List[str] = []
        
    def evaluate_model(self, max_samples: Optional[int] = None):
        """Evaluate model trên test set và collect predictions"""
        print("\n🔍 Evaluating model on test set...")
        
        samples = self.test_data[:max_samples] if max_samples else self.test_data
        
        for i, sample in enumerate(samples):
            if (i + 1) % 100 == 0:
                print(f"  Processing {i+1}/{len(samples)} samples...")
            
            text = sample.get("input", "")
            if not text:
                continue
            
            # Get ground truth
            intent_label = sample.get("intent", "unknown")
            command_label = sample.get("command", "unknown")
            
            # Convert to IDs
            intent_id = self.intent_label2id.get(intent_label, 0)
            command_id = self.command_label2id.get(command_label, 0)
            
            # Get ground truth entities (from BIO tags)
            bio_labels = sample.get("bio_labels", [])
            if bio_labels:
                entity_seq = [self.entity_id2label.get(id, "O") for id in bio_labels]
            else:
                # Extract from entities dict
                entities = sample.get("entities", [])
                entity_seq = self._extract_entity_sequence(text, entities)
            
            # Predict
            try:
                result = self.model_inference.predict(text)
                
                # Intent prediction
                pred_intent = result.get("intent", "unknown")
                pred_intent_id = self.intent_label2id.get(pred_intent, 0)
                
                # Command prediction
                pred_command = result.get("command", "unknown")
                pred_command_id = self.command_label2id.get(pred_command, 0)
                
                # Entity prediction (from model result)
                pred_entities = result.get("entities", {})
                pred_entity_seq = self._extract_entity_sequence(text, pred_entities)
                
                # Store
                self.texts.append(text)
                self.intent_labels.append(intent_id)
                self.intent_predictions.append(pred_intent_id)
                self.command_labels.append(command_id)
                self.command_predictions.append(pred_command_id)
                self.entity_labels.append(entity_seq)
                self.entity_predictions.append(pred_entity_seq)
                
                # Also store entity dicts for detailed analysis
                if not hasattr(self, 'true_entity_dicts'):
                    self.true_entity_dicts: List[Dict[str, List[str]]] = []
                    self.pred_entity_dicts: List[Dict[str, List[str]]] = []
                
                true_entities_list = sample.get("entities", [])
                self.true_entity_dicts.append(self._extract_entities_dict(text, true_entities_list))
                self.pred_entity_dicts.append(self._extract_entities_dict(text, pred_entities))
                
            except Exception as e:
                print(f"⚠️ Error predicting for '{text}': {e}")
                continue
        
        print(f"✅ Evaluation complete: {len(self.intent_predictions)} samples processed")
    
    def _extract_entity_sequence(self, text: str, entities: Any) -> List[str]:
        """Extract entity sequence from entities dict/list"""
        # Initialize with O tags
        tokens = text.split()
        entity_seq = ["O"] * len(tokens)
        
        if isinstance(entities, dict):
            # Entities dict: {ENTITY_TYPE: value}
            for entity_type, value in entities.items():
                if isinstance(value, str) and value:
                    # Simple matching (can be improved)
                    value_tokens = value.split()
                    for i, token in enumerate(tokens):
                        if token in value_tokens:
                            if i == 0 or entity_seq[i-1] == "O" or not entity_seq[i-1].startswith("I-"):
                                entity_seq[i] = f"B-{entity_type}"
                            else:
                                entity_seq[i] = f"I-{entity_type}"
        elif isinstance(entities, list):
            # Entities list: [{"label": "ENTITY_TYPE", "text": "...", "start": ..., "end": ...}]
            for entity in entities:
                if isinstance(entity, dict):
                    entity_type = entity.get("label", "UNKNOWN")
                    entity_text = entity.get("text", "")
                    start = entity.get("start", 0)
                    end = entity.get("end", len(text))
                    
                    # Map to tokens
                    entity_tokens = entity_text.split()
                    text_before = text[:start]
                    token_start = len(text_before.split())
                    
                    for j, token in enumerate(entity_tokens):
                        idx = token_start + j
                        if idx < len(entity_seq):
                            if j == 0:
                                entity_seq[idx] = f"B-{entity_type}"
                            else:
                                entity_seq[idx] = f"I-{entity_type}"
        
        return entity_seq
    
    def _extract_entities_dict(self, text: str, entities: Any) -> Dict[str, List[str]]:
        """Extract entities as dict grouped by entity type"""
        entity_dict: Dict[str, List[str]] = defaultdict(list)
        
        if isinstance(entities, dict):
            # Entities dict: {ENTITY_TYPE: value}
            for entity_type, value in entities.items():
                if isinstance(value, str) and value.strip():
                    entity_dict[entity_type].append(value.strip())
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, str) and v.strip():
                            entity_dict[entity_type].append(v.strip())
        elif isinstance(entities, list):
            # Entities list: [{"label": "ENTITY_TYPE", "text": "...", ...}]
            for entity in entities:
                if isinstance(entity, dict):
                    entity_type = entity.get("label", "UNKNOWN")
                    entity_text = entity.get("text", "")
                    if entity_text and entity_text.strip():
                        entity_dict[entity_type].append(entity_text.strip())
        
        return dict(entity_dict)
    
    def _extract_entities_dict(self, text: str, entities: Any) -> Dict[str, List[str]]:
        """Extract entities as dict grouped by entity type"""
        entity_dict: Dict[str, List[str]] = defaultdict(list)
        
        if isinstance(entities, dict):
            # Entities dict: {ENTITY_TYPE: value}
            for entity_type, value in entities.items():
                if isinstance(value, str) and value.strip():
                    entity_dict[entity_type].append(value.strip())
        elif isinstance(entities, list):
            # Entities list: [{"label": "ENTITY_TYPE", "text": "...", ...}]
            for entity in entities:
                if isinstance(entity, dict):
                    entity_type = entity.get("label", "UNKNOWN")
                    entity_text = entity.get("text", "")
                    if entity_text and entity_text.strip():
                        entity_dict[entity_type].append(entity_text.strip())
        
        return dict(entity_dict)
    
    def visualize_intent_classification(self, output_dir: Path):
        """Visualize Intent Classification results"""
        print("\n📊 Visualizing Intent Classification...")
        
        # Get label names
        intent_label_names = [self.intent_id2label.get(i, f"intent_{i}") for i in range(len(self.intent_config.intent_labels))]
        
        # Calculate metrics
        num_intents = len(intent_label_names)
        accuracy = accuracy_score(self.intent_labels, self.intent_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.intent_labels, self.intent_predictions, 
            labels=range(num_intents), 
            average=None, 
            zero_division=0
        )
        macro_f1 = f1_score(self.intent_labels, self.intent_predictions, average='macro', zero_division=0)
        weighted_f1 = f1_score(self.intent_labels, self.intent_predictions, average='weighted', zero_division=0)
        
        # 1. Confusion Matrix
        fig, ax = plt.subplots(figsize=(14, 12))
        cm = confusion_matrix(self.intent_labels, self.intent_predictions, labels=range(num_intents))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=intent_label_names,
            yticklabels=intent_label_names,
            ax=ax,
            cbar_kws={'label': 'Normalized Count'}
        )
        ax.set_title('Intent Classification - Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Intent', fontsize=12)
        ax.set_ylabel('True Intent', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'intent_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-Class Metrics Bar Chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        x = np.arange(len(intent_label_names))
        width = 0.25
        
        # Precision, Recall, F1
        axes[0, 0].bar(x - width, precision, width, label='Precision', alpha=0.8)
        axes[0, 0].bar(x, recall, width, label='Recall', alpha=0.8)
        axes[0, 0].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        axes[0, 0].set_xlabel('Intent', fontsize=11)
        axes[0, 0].set_ylabel('Score', fontsize=11)
        axes[0, 0].set_title('Intent Classification - Per-Class Metrics', fontsize=13, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(intent_label_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0, 1.1])
        
        # Support (number of samples)
        axes[0, 1].bar(x, support, alpha=0.7, color='steelblue')
        axes[0, 1].set_xlabel('Intent', fontsize=11)
        axes[0, 1].set_ylabel('Number of Samples', fontsize=11)
        axes[0, 1].set_title('Intent Classification - Support (Number of Samples)', fontsize=13, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(intent_label_names, rotation=45, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # F1-Score comparison
        axes[1, 0].barh(x, f1, alpha=0.7, color='green')
        axes[1, 0].set_yticks(x)
        axes[1, 0].set_yticklabels(intent_label_names)
        axes[1, 0].set_xlabel('F1-Score', fontsize=11)
        axes[1, 0].set_title('Intent Classification - F1-Score per Class', fontsize=13, fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        axes[1, 0].set_xlim([0, 1.1])
        
        # Overall metrics summary
        metrics_summary = {
            'Accuracy': accuracy,
            'Macro F1': macro_f1,
            'Weighted F1': weighted_f1
        }
        axes[1, 1].bar(metrics_summary.keys(), metrics_summary.values(), alpha=0.7, color='orange')
        axes[1, 1].set_ylabel('Score', fontsize=11)
        axes[1, 1].set_title('Intent Classification - Overall Metrics', fontsize=13, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].set_ylim([0, 1.1])
        for i, (k, v) in enumerate(metrics_summary.items()):
            axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'intent_classification_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Classification Report
        report = classification_report(
            self.intent_labels, self.intent_predictions,
            target_names=intent_label_names,
            output_dict=True,
            zero_division=0
        )
        
        print(f"\n📋 Intent Classification Report:")
        print(f"   Overall Accuracy: {accuracy:.4f}")
        print(f"   Macro F1: {macro_f1:.4f}")
        print(f"   Weighted F1: {weighted_f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class': {
                label: {
                    'precision': float(p),
                    'recall': float(r),
                    'f1': float(f),
                    'support': int(s)
                }
                for label, p, r, f, s in zip(intent_label_names, precision, recall, f1, support)
            }
        }
    
    def visualize_command_classification(self, output_dir: Path):
        """Visualize Command Classification results"""
        print("\n📊 Visualizing Command Classification...")
        
        # Get label names
        command_label_names = [self.command_id2label.get(i, f"command_{i}") for i in range(len(self.command_config.command_labels))]
        
        # Calculate metrics
        num_commands = len(command_label_names)
        accuracy = accuracy_score(self.command_labels, self.command_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.command_labels, self.command_predictions,
            labels=range(num_commands),
            average=None,
            zero_division=0
        )
        macro_f1 = f1_score(self.command_labels, self.command_predictions, average='macro', zero_division=0)
        weighted_f1 = f1_score(self.command_labels, self.command_predictions, average='weighted', zero_division=0)
        
        # 1. Confusion Matrix
        fig, ax = plt.subplots(figsize=(14, 12))
        cm = confusion_matrix(self.command_labels, self.command_predictions, labels=range(num_commands))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Oranges',
            xticklabels=command_label_names,
            yticklabels=command_label_names,
            ax=ax,
            cbar_kws={'label': 'Normalized Count'}
        )
        ax.set_title('Command Classification - Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Command', fontsize=12)
        ax.set_ylabel('True Command', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'command_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-Class Metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        x = np.arange(len(command_label_names))
        width = 0.25
        
        axes[0, 0].bar(x - width, precision, width, label='Precision', alpha=0.8)
        axes[0, 0].bar(x, recall, width, label='Recall', alpha=0.8)
        axes[0, 0].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        axes[0, 0].set_xlabel('Command', fontsize=11)
        axes[0, 0].set_ylabel('Score', fontsize=11)
        axes[0, 0].set_title('Command Classification - Per-Class Metrics', fontsize=13, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(command_label_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0, 1.1])
        
        axes[0, 1].bar(x, support, alpha=0.7, color='steelblue')
        axes[0, 1].set_xlabel('Command', fontsize=11)
        axes[0, 1].set_ylabel('Number of Samples', fontsize=11)
        axes[0, 1].set_title('Command Classification - Support', fontsize=13, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(command_label_names, rotation=45, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        axes[1, 0].barh(x, f1, alpha=0.7, color='green')
        axes[1, 0].set_yticks(x)
        axes[1, 0].set_yticklabels(command_label_names)
        axes[1, 0].set_xlabel('F1-Score', fontsize=11)
        axes[1, 0].set_title('Command Classification - F1-Score per Class', fontsize=13, fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        axes[1, 0].set_xlim([0, 1.1])
        
        metrics_summary = {
            'Accuracy': accuracy,
            'Macro F1': macro_f1,
            'Weighted F1': weighted_f1
        }
        axes[1, 1].bar(metrics_summary.keys(), metrics_summary.values(), alpha=0.7, color='orange')
        axes[1, 1].set_ylabel('Score', fontsize=11)
        axes[1, 1].set_title('Command Classification - Overall Metrics', fontsize=13, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].set_ylim([0, 1.1])
        for i, (k, v) in enumerate(metrics_summary.items()):
            axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'command_classification_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📋 Command Classification Report:")
        print(f"   Overall Accuracy: {accuracy:.4f}")
        print(f"   Macro F1: {macro_f1:.4f}")
        print(f"   Weighted F1: {weighted_f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class': {
                label: {
                    'precision': float(p),
                    'recall': float(r),
                    'f1': float(f),
                    'support': int(s)
                }
                for label, p, r, f, s in zip(command_label_names, precision, recall, f1, support)
            }
        }
    
    def visualize_ner_results(self, output_dir: Path):
        """Visualize NER results"""
        print("\n📊 Visualizing NER Results...")
        
        # Calculate seqeval metrics
        try:
            ner_f1 = seqeval_f1(self.entity_labels, self.entity_predictions)
            ner_precision = seqeval_precision(self.entity_labels, self.entity_predictions)
            ner_recall = seqeval_recall(self.entity_labels, self.entity_predictions)
            
            # Get classification report
            report = seqeval_classification_report(self.entity_labels, self.entity_predictions, output_dict=True)
            
            print(f"\n📋 NER Report:")
            print(f"   Precision: {ner_precision:.4f}")
            print(f"   Recall: {ner_recall:.4f}")
            print(f"   F1-Score: {ner_f1:.4f}")
            
            # Extract entity types from report
            entity_types = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
            support_counts = []
            
            for label, metrics in report.items():
                if label not in ['micro avg', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
                    entity_types.append(label)
                    precision_scores.append(metrics.get('precision', 0))
                    recall_scores.append(metrics.get('recall', 0))
                    f1_scores.append(metrics.get('f1-score', 0))
                    support_counts.append(metrics.get('support', 0))
            
            if entity_types:
                # Per-Entity-Type Metrics
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                x = np.arange(len(entity_types))
                width = 0.25
                
                axes[0, 0].bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
                axes[0, 0].bar(x, recall_scores, width, label='Recall', alpha=0.8)
                axes[0, 0].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
                axes[0, 0].set_xlabel('Entity Type', fontsize=11)
                axes[0, 0].set_ylabel('Score', fontsize=11)
                axes[0, 0].set_title('NER - Per-Entity-Type Metrics', fontsize=13, fontweight='bold')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(entity_types, rotation=45, ha='right')
                axes[0, 0].legend()
                axes[0, 0].grid(axis='y', alpha=0.3)
                axes[0, 0].set_ylim([0, 1.1])
                
                axes[0, 1].bar(x, support_counts, alpha=0.7, color='steelblue')
                axes[0, 1].set_xlabel('Entity Type', fontsize=11)
                axes[0, 1].set_ylabel('Number of Entities', fontsize=11)
                axes[0, 1].set_title('NER - Support (Number of Entities)', fontsize=13, fontweight='bold')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(entity_types, rotation=45, ha='right')
                axes[0, 1].grid(axis='y', alpha=0.3)
                
                axes[1, 0].barh(x, f1_scores, alpha=0.7, color='green')
                axes[1, 0].set_yticks(x)
                axes[1, 0].set_yticklabels(entity_types)
                axes[1, 0].set_xlabel('F1-Score', fontsize=11)
                axes[1, 0].set_title('NER - F1-Score per Entity Type', fontsize=13, fontweight='bold')
                axes[1, 0].grid(axis='x', alpha=0.3)
                axes[1, 0].set_xlim([0, 1.1])
                
                metrics_summary = {
                    'Precision': ner_precision,
                    'Recall': ner_recall,
                    'F1-Score': ner_f1
                }
                axes[1, 1].bar(metrics_summary.keys(), metrics_summary.values(), alpha=0.7, color='purple')
                axes[1, 1].set_ylabel('Score', fontsize=11)
                axes[1, 1].set_title('NER - Overall Metrics', fontsize=13, fontweight='bold')
                axes[1, 1].grid(axis='y', alpha=0.3)
                axes[1, 1].set_ylim([0, 1.1])
                for i, (k, v) in enumerate(metrics_summary.items()):
                    axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'ner_metrics.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            return {
                'precision': float(ner_precision),
                'recall': float(ner_recall),
                'f1': float(ner_f1),
                'per_entity_type': {
                    label: {
                        'precision': float(metrics.get('precision', 0)),
                        'recall': float(metrics.get('recall', 0)),
                        'f1': float(metrics.get('f1-score', 0)),
                        'support': int(metrics.get('support', 0))
                    }
                    for label, metrics in report.items()
                    if label not in ['micro avg', 'macro avg', 'weighted avg'] and isinstance(metrics, dict)
                }
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating NER metrics: {e}")
            return {}
    
    def create_sample_predictions_html(self, output_dir: Path, num_samples: int = 20):
        """Create HTML file với sample predictions và highlighted entities"""
        print(f"\n📝 Creating sample predictions HTML...")
        
        # Select diverse samples (mix of correct and incorrect predictions)
        sample_indices = list(range(min(num_samples, len(self.texts))))
        
        html_content = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Predictions - Sample Results</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .sample {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .text {
            font-size: 16px;
            margin: 10px 0;
            padding: 10px;
            background: #f9f9f9;
            border-left: 4px solid #007bff;
        }
        .prediction {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 10px 0;
        }
        .intent, .command {
            padding: 10px;
            border-radius: 4px;
        }
        .correct {
            background: #d4edda;
            border-left: 4px solid #28a745;
        }
        .incorrect {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
        }
        .label {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .entities {
            margin: 10px 0;
        }
        .entity {
            display: inline-block;
            padding: 2px 8px;
            margin: 2px;
            border-radius: 4px;
            font-size: 14px;
        }
        .entity-pred {
            background: #cfe2ff;
            border: 1px solid #0d6efd;
        }
        .entity-true {
            background: #d1e7dd;
            border: 1px solid #198754;
        }
    </style>
</head>
<body>
    <h1>🤖 Model Predictions - Sample Results</h1>
"""
        
        intent_label_names = [self.intent_id2label.get(i, f"intent_{i}") for i in range(len(self.intent_config.intent_labels))]
        command_label_names = [self.command_id2label.get(i, f"command_{i}") for i in range(len(self.command_config.command_labels))]
        
        for idx in sample_indices:
            if idx >= len(self.texts):
                continue
            
            text = self.texts[idx]
            true_intent = intent_label_names[self.intent_labels[idx]] if self.intent_labels[idx] < len(intent_label_names) else "unknown"
            pred_intent = intent_label_names[self.intent_predictions[idx]] if self.intent_predictions[idx] < len(intent_label_names) else "unknown"
            true_command = command_label_names[self.command_labels[idx]] if self.command_labels[idx] < len(command_label_names) else "unknown"
            pred_command = command_label_names[self.command_predictions[idx]] if self.command_predictions[idx] < len(command_label_names) else "unknown"
            
            intent_correct = true_intent == pred_intent
            command_correct = true_command == pred_command
            
            html_content += f"""
    <div class="sample">
        <div class="text"><strong>Input:</strong> {text}</div>
        <div class="prediction">
            <div class="intent {'correct' if intent_correct else 'incorrect'}">
                <div class="label">Intent:</div>
                <div>True: <strong>{true_intent}</strong></div>
                <div>Pred: <strong>{pred_intent}</strong></div>
            </div>
            <div class="command {'correct' if command_correct else 'incorrect'}">
                <div class="label">Command:</div>
                <div>True: <strong>{true_command}</strong></div>
                <div>Pred: <strong>{pred_command}</strong></div>
            </div>
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        output_file = output_dir / 'sample_predictions.html'
        output_file.write_text(html_content, encoding='utf-8')
        print(f"✅ Saved sample predictions HTML to: {output_file}")
    
    def generate_report(self, output_dir: Path):
        """Generate comprehensive visualization report"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("🎨 Generating Model Results Visualization Report")
        print("="*60)
        
        # Evaluate model
        self.evaluate_model()
        
        # Visualize Intent Classification
        intent_metrics = self.visualize_intent_classification(output_dir)
        
        # Visualize Command Classification
        command_metrics = self.visualize_command_classification(output_dir)
        
        # Visualize NER
        ner_metrics = self.visualize_ner_results(output_dir)
        
        # Create sample predictions HTML
        self.create_sample_predictions_html(output_dir, num_samples=30)
        
        # Save metrics JSON
        report = {
            'intent_classification': intent_metrics,
            'command_classification': command_metrics,
            'ner': ner_metrics,
            'num_samples': len(self.texts)
        }
        
        report_file = output_dir / 'evaluation_metrics.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Report saved to: {output_dir}")
        print(f"   - Intent Confusion Matrix: {output_dir / 'intent_confusion_matrix.png'}")
        print(f"   - Intent Metrics: {output_dir / 'intent_classification_metrics.png'}")
        print(f"   - Command Confusion Matrix: {output_dir / 'command_confusion_matrix.png'}")
        print(f"   - Command Metrics: {output_dir / 'command_classification_metrics.png'}")
        print(f"   - NER Metrics: {output_dir / 'ner_metrics.png'}")
        print(f"   - Sample Predictions: {output_dir / 'sample_predictions.html'}")
        print(f"   - Metrics JSON: {output_dir / 'evaluation_metrics.json'}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Intent/Command Classification and NER Results')
    parser.add_argument('--model-path', type=str, default='models/phobert_multitask',
                        help='Path to trained model directory')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Path to test data JSON file (default: from config)')
    parser.add_argument('--output-dir', type=str, default='reports/model_results',
                        help='Output directory for visualizations')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    visualizer = ModelResultsVisualizer(
        model_path=args.model_path,
        test_data_path=args.test_data
    )
    
    visualizer.generate_report(Path(args.output_dir))


if __name__ == '__main__':
    main()

