#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed NER Visualization - Đánh giá chi tiết cho từng Entity Type
Bổ sung cho visualize_model_results.py
"""

import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_loader import TrainedModelInference
from src.data.processed.data_processor import DataProcessor
from src.training.configs.config import EntityConfig, ModelConfig

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11


class DetailedNERVisualizer:
    """Detailed NER Visualizer cho từng entity type"""
    
    def __init__(self, model_path: str = "models/phobert_multitask", test_data_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path) if test_data_path else Path(ModelConfig().test_data_path)
        
        print(f"📦 Loading model from: {self.model_path}")
        self.model_inference = TrainedModelInference(str(self.model_path))
        
        self.entity_config = EntityConfig()
        self.processor = DataProcessor()
        
        print(f"📚 Loading test data from: {self.test_data_path}")
        self.test_data = self.processor.load_dataset(str(self.test_data_path))
        print(f"✅ Loaded {len(self.test_data)} test samples")
        
        # Storage
        self.true_entity_dicts: List[Dict[str, List[str]]] = []
        self.pred_entity_dicts: List[Dict[str, List[str]]] = []
        self.texts: List[str] = []
    
    def _extract_entities_dict(self, text: str, entities: Any) -> Dict[str, List[str]]:
        """Extract entities as dict grouped by entity type"""
        entity_dict: Dict[str, List[str]] = defaultdict(list)
        
        if isinstance(entities, dict):
            for entity_type, value in entities.items():
                if isinstance(value, str) and value.strip():
                    entity_dict[entity_type].append(value.strip())
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, str) and v.strip():
                            entity_dict[entity_type].append(v.strip())
        elif isinstance(entities, list):
            for entity in entities:
                if isinstance(entity, dict):
                    entity_type = entity.get("label", "UNKNOWN")
                    entity_text = entity.get("text", "")
                    if entity_text and entity_text.strip():
                        entity_dict[entity_type].append(entity_text.strip())
        
        return dict(entity_dict)
    
    def evaluate_entities(self, max_samples: Optional[int] = None):
        """Evaluate entity extraction"""
        print("\n🔍 Evaluating Entity Extraction...")
        
        samples = self.test_data[:max_samples] if max_samples else self.test_data
        
        for i, sample in enumerate(samples):
            if (i + 1) % 100 == 0:
                print(f"  Processing {i+1}/{len(samples)} samples...")
            
            text = sample.get("input", "")
            if not text:
                continue
            
            # Get ground truth entities
            true_entities = sample.get("entities", [])
            true_entity_dict = self._extract_entities_dict(text, true_entities)
            
            # Predict
            try:
                result = self.model_inference.predict(text)
                pred_entities = result.get("entities", {})
                pred_entity_dict = self._extract_entities_dict(text, pred_entities)
                
                self.texts.append(text)
                self.true_entity_dicts.append(true_entity_dict)
                self.pred_entity_dicts.append(pred_entity_dict)
                
            except Exception as e:
                print(f"⚠️ Error: {e}")
                continue
        
        print(f"✅ Evaluation complete: {len(self.texts)} samples processed")
    
    def calculate_per_entity_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate detailed metrics per entity type"""
        all_entity_types = set()
        for entity_dict in self.true_entity_dicts + self.pred_entity_dicts:
            all_entity_types.update(entity_dict.keys())
        
        metrics_per_type: Dict[str, Dict[str, Any]] = {}
        
        for entity_type in all_entity_types:
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            total_true = 0
            total_pred = 0
            
            for true_dict, pred_dict in zip(self.true_entity_dicts, self.pred_entity_dicts):
                true_values = set(true_dict.get(entity_type, []))
                pred_values = set(pred_dict.get(entity_type, []))
                
                total_true += len(true_values)
                total_pred += len(pred_values)
                
                # Exact match
                tp = len(true_values & pred_values)
                fp = len(pred_values - true_values)
                fn = len(true_values - pred_values)
                
                true_positives += tp
                false_positives += fp
                false_negatives += fn
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = true_positives / total_true if total_true > 0 else 0.0
            
            metrics_per_type[entity_type] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy),
                'support': int(total_true),
                'predicted': int(total_pred),
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives)
            }
        
        return metrics_per_type
    
    def visualize_detailed_ner(self, output_dir: Path):
        """Create detailed NER visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("🎨 Generating Detailed NER Visualization")
        print("="*60)
        
        # Evaluate
        self.evaluate_entities()
        
        # Calculate metrics
        metrics = self.calculate_per_entity_metrics()
        
        if not metrics:
            print("⚠️ No entity metrics to visualize")
            return
        
        # Sort by support (descending)
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1]['support'], reverse=True)
        entity_types = [item[0] for item in sorted_metrics]
        
        precision_scores = [metrics[et]['precision'] for et in entity_types]
        recall_scores = [metrics[et]['recall'] for et in entity_types]
        f1_scores = [metrics[et]['f1'] for et in entity_types]
        accuracy_scores = [metrics[et]['accuracy'] for et in entity_types]
        support_counts = [metrics[et]['support'] for et in entity_types]
        predicted_counts = [metrics[et]['predicted'] for et in entity_types]
        
        # 1. Comprehensive Metrics Chart (4 subplots)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        x = np.arange(len(entity_types))
        width = 0.2
        
        # Precision, Recall, F1, Accuracy
        axes[0, 0].bar(x - 1.5*width, precision_scores, width, label='Precision', alpha=0.9, color='#3498db')
        axes[0, 0].bar(x - 0.5*width, recall_scores, width, label='Recall', alpha=0.9, color='#e74c3c')
        axes[0, 0].bar(x + 0.5*width, f1_scores, width, label='F1-Score', alpha=0.9, color='#2ecc71')
        axes[0, 0].bar(x + 1.5*width, accuracy_scores, width, label='Accuracy', alpha=0.9, color='#f39c12')
        axes[0, 0].set_xlabel('Entity Type', fontsize=13, fontweight='bold')
        axes[0, 0].set_ylabel('Score', fontsize=13, fontweight='bold')
        axes[0, 0].set_title('NER - Comprehensive Metrics per Entity Type', fontsize=15, fontweight='bold', pad=20)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(entity_types, rotation=45, ha='right', fontsize=10)
        axes[0, 0].legend(fontsize=11, loc='upper right')
        axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
        axes[0, 0].set_ylim([0, 1.15])
        
        # Add value labels
        for i, (p, r, f, a) in enumerate(zip(precision_scores, recall_scores, f1_scores, accuracy_scores)):
            if p > 0: axes[0, 0].text(i - 1.5*width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=7)
            if r > 0: axes[0, 0].text(i - 0.5*width, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=7)
            if f > 0: axes[0, 0].text(i + 0.5*width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=7)
            if a > 0: axes[0, 0].text(i + 1.5*width, a + 0.02, f'{a:.2f}', ha='center', va='bottom', fontsize=7)
        
        # Support vs Predicted
        x_support = np.arange(len(entity_types))
        width_support = 0.35
        axes[0, 1].bar(x_support - width_support/2, support_counts, width_support, label='True (Ground Truth)', alpha=0.8, color='#3498db')
        axes[0, 1].bar(x_support + width_support/2, predicted_counts, width_support, label='Predicted', alpha=0.8, color='#e74c3c')
        axes[0, 1].set_xlabel('Entity Type', fontsize=13, fontweight='bold')
        axes[0, 1].set_ylabel('Count', fontsize=13, fontweight='bold')
        axes[0, 1].set_title('Entity Count: True vs Predicted', fontsize=15, fontweight='bold', pad=20)
        axes[0, 1].set_xticks(x_support)
        axes[0, 1].set_xticklabels(entity_types, rotation=45, ha='right', fontsize=10)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (true_count, pred_count) in enumerate(zip(support_counts, predicted_counts)):
            max_height = max(true_count, pred_count)
            if true_count > 0:
                axes[0, 1].text(i - width_support/2, true_count + max_height*0.02, f'{int(true_count)}', 
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
            if pred_count > 0:
                axes[0, 1].text(i + width_support/2, pred_count + max_height*0.02, f'{int(pred_count)}', 
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # F1-Score Ranking
        colors_f1 = ['#2ecc71' if f >= 0.7 else '#f39c12' if f >= 0.5 else '#e74c3c' for f in f1_scores]
        axes[1, 0].barh(x, f1_scores, alpha=0.9, color=colors_f1)
        axes[1, 0].set_yticks(x)
        axes[1, 0].set_yticklabels(entity_types, fontsize=10)
        axes[1, 0].set_xlabel('F1-Score', fontsize=13, fontweight='bold')
        axes[1, 0].set_title('F1-Score Ranking per Entity Type', fontsize=15, fontweight='bold', pad=20)
        axes[1, 0].grid(axis='x', alpha=0.3, linestyle='--')
        axes[1, 0].set_xlim([0, 1.15])
        
        # Add value labels and threshold lines
        axes[1, 0].axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Good (≥0.7)')
        axes[1, 0].axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (≥0.5)')
        axes[1, 0].legend(fontsize=10, loc='lower right')
        
        for i, (y_pos, f1) in enumerate(zip(x, f1_scores)):
            if f1 > 0:
                axes[1, 0].text(f1 + 0.02, y_pos, f'{f1:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # Error Analysis (FP, FN)
        fp_counts = [metrics[et]['false_positives'] for et in entity_types]
        fn_counts = [metrics[et]['false_negatives'] for et in entity_types]
        tp_counts = [metrics[et]['true_positives'] for et in entity_types]
        
        x_error = np.arange(len(entity_types))
        width_error = 0.25
        axes[1, 1].bar(x_error - width_error, tp_counts, width_error, label='True Positives', alpha=0.9, color='#2ecc71')
        axes[1, 1].bar(x_error, fp_counts, width_error, label='False Positives', alpha=0.9, color='#e74c3c')
        axes[1, 1].bar(x_error + width_error, fn_counts, width_error, label='False Negatives', alpha=0.9, color='#f39c12')
        axes[1, 1].set_xlabel('Entity Type', fontsize=13, fontweight='bold')
        axes[1, 1].set_ylabel('Count', fontsize=13, fontweight='bold')
        axes[1, 1].set_title('Error Analysis: TP, FP, FN per Entity Type', fontsize=15, fontweight='bold', pad=20)
        axes[1, 1].set_xticks(x_error)
        axes[1, 1].set_xticklabels(entity_types, rotation=45, ha='right', fontsize=10)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ner_detailed_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Entity Distribution Chart
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Distribution in dataset
        entity_distribution = Counter()
        for entity_dict in self.true_entity_dicts:
            for entity_type, values in entity_dict.items():
                entity_distribution[entity_type] += len(values)
        
        if entity_distribution:
            dist_types = [et for et in entity_types if et in entity_distribution]
            dist_counts = [entity_distribution[et] for et in dist_types]
            
            colors_dist = plt.cm.viridis(np.linspace(0, 1, len(dist_types)))
            bars = axes[0].bar(range(len(dist_types)), dist_counts, alpha=0.8, color=colors_dist)
            axes[0].set_xlabel('Entity Type', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Total Count in Test Set', fontsize=12, fontweight='bold')
            axes[0].set_title('Entity Distribution in Test Dataset', fontsize=14, fontweight='bold', pad=15)
            axes[0].set_xticks(range(len(dist_types)))
            axes[0].set_xticklabels(dist_types, rotation=45, ha='right', fontsize=10)
            axes[0].grid(axis='y', alpha=0.3, linestyle='--')
            
            for i, (bar, count) in enumerate(zip(bars, dist_counts)):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + max(dist_counts)*0.01,
                           f'{int(count)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Pie chart for top entities
        if entity_distribution:
            top_entities = entity_distribution.most_common(10)
            top_types = [item[0] for item in top_entities]
            top_counts = [item[1] for item in top_entities]
            
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(top_types)))
            wedges, texts, autotexts = axes[1].pie(top_counts, labels=top_types, autopct='%1.1f%%',
                                                   colors=colors_pie, startangle=90, textprops={'fontsize': 9})
            axes[1].set_title('Top 10 Entity Types Distribution', fontsize=14, fontweight='bold', pad=15)
            
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ner_entity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Per-Entity-Type Detailed Table
        print("\n📊 Per-Entity-Type Metrics:")
        print("="*80)
        print(f"{'Entity Type':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12} {'Support':<10}")
        print("="*80)
        
        for entity_type in entity_types:
            m = metrics[entity_type]
            print(f"{entity_type:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['accuracy']:<12.4f} {m['support']:<10}")
        
        # Save metrics to JSON
        report = {
            'per_entity_type': metrics,
            'summary': {
                'total_entity_types': len(entity_types),
                'total_samples': len(self.texts)
            }
        }
        
        report_file = output_dir / 'ner_detailed_metrics.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Detailed NER visualization saved to: {output_dir}")
        print(f"   - Comprehensive Metrics: {output_dir / 'ner_detailed_metrics.png'}")
        print(f"   - Entity Distribution: {output_dir / 'ner_entity_distribution.png'}")
        print(f"   - Metrics JSON: {output_dir / 'ner_detailed_metrics.json'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Detailed NER Visualization')
    parser.add_argument('--model-path', type=str, default='models/phobert_multitask',
                        help='Path to trained model directory')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Path to test data JSON file')
    parser.add_argument('--output-dir', type=str, default='reports/ner_detailed',
                        help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples')
    
    args = parser.parse_args()
    
    visualizer = DetailedNERVisualizer(
        model_path=args.model_path,
        test_data_path=args.test_data
    )
    
    visualizer.visualize_detailed_ner(Path(args.output_dir))


if __name__ == '__main__':
    main()

