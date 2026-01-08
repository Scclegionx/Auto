#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Script for Hybrid System Intervention Analysis
Tạo các biểu đồ trực quan để trình bày mức độ can thiệp của Hybrid System
"""

import sys
from pathlib import Path
import json
from typing import Dict, List, Any
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class HybridInterventionVisualizer:
    """Tạo visualizations cho Hybrid System Intervention Analysis"""
    
    def __init__(self, report_path: Path):
        self.report_path = report_path
        self.report_data = self._load_report()
        
    def _load_report(self) -> Dict[str, Any]:
        """Load intervention report"""
        with open(self.report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_comprehensive_dashboard(self, output_dir: Path):
        """Tạo comprehensive dashboard với tất cả visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("[*] Creating comprehensive intervention dashboard...")
        
        # 1. Main Overview Dashboard
        self._create_main_dashboard(output_dir)
        
        # 2. Intervention Level Analysis
        self._create_intervention_level_chart(output_dir)
        
        # 3. Improvement vs Degradation
        self._create_improvement_chart(output_dir)
        
        # 4. Accuracy Comparison
        self._create_accuracy_comparison(output_dir)
        
        # 5. Decision Reasons
        self._create_decision_reasons_chart(output_dir)
        
        # 6. Confidence Changes
        self._create_confidence_analysis(output_dir)
        
        print(f"[OK] Dashboard saved to: {output_dir}")
    
    def _create_main_dashboard(self, output_dir: Path):
        """Tạo main dashboard với overview"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        summary = self.report_data['summary']
        improvements = self.report_data['improvements']
        accuracy = self.report_data['accuracy']
        decision_reasons = self.report_data['decision_reasons']
        
        total = summary['total_samples']
        
        # 1. Intervention Statistics (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        intervention_types = ['Intent\nChanged', 'Command\nChanged', 'Confidence\nChanged', 'Entities\nChanged']
        intervention_counts = [
            summary['intent_changed'],
            summary['command_changed'],
            summary['confidence_changed'],
            summary['entities_changed']
        ]
        intervention_pct = [c/total*100 for c in intervention_counts]
        colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
        bars = ax1.bar(intervention_types, intervention_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Number of Changes', fontsize=12, fontweight='bold')
        ax1.set_title('Intervention Frequency', fontsize=13, fontweight='bold', pad=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, (bar, pct) in enumerate(zip(bars, intervention_pct)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(intervention_counts)*0.02,
                    f'{int(intervention_counts[i])}\n({pct:.1f}%)', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        
        # 2. Improvement vs Degradation (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        improve_data = {
            'Intent\nImproved': improvements['intent_improved'],
            'Intent\nDegraded': improvements['intent_degraded'],
            'Command\nImproved': improvements['command_improved'],
            'Command\nDegraded': improvements['command_degraded']
        }
        colors_improve = ['#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c']
        x = np.arange(len(improve_data))
        bars2 = ax2.bar(x, list(improve_data.values()), color=colors_improve, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(list(improve_data.keys()), fontsize=10)
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('Improvement vs Degradation', fontsize=13, fontweight='bold', pad=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars2, improve_data.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(improve_data.values())*0.02,
                    f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Accuracy Comparison (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        acc_data = {
            'Raw\nIntent': accuracy['raw_intent_accuracy'],
            'Hybrid\nIntent': accuracy['hybrid_intent_accuracy'],
            'Raw\nCommand': accuracy['raw_command_accuracy'],
            'Hybrid\nCommand': accuracy['hybrid_command_accuracy']
        }
        colors_acc = ['#95a5a6', '#3498db', '#95a5a6', '#3498db']
        x_acc = np.arange(len(acc_data))
        bars3 = ax3.bar(x_acc, list(acc_data.values()), color=colors_acc, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_xticks(x_acc)
        ax3.set_xticklabels(list(acc_data.keys()), fontsize=10)
        ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title('Accuracy Comparison', fontsize=13, fontweight='bold', pad=10)
        ax3.set_ylim([0, 1.1])
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars3, acc_data.values()):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. Decision Reasons (Middle Left - Large)
        ax4 = fig.add_subplot(gs[1:, 0])
        top_reasons = dict(sorted(decision_reasons.items(), key=lambda x: x[1], reverse=True)[:8])
        if top_reasons:
            x_reasons = np.arange(len(top_reasons))
            bars4 = ax4.barh(x_reasons, list(top_reasons.values()), color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)
            ax4.set_yticks(x_reasons)
            ax4.set_yticklabels([r.replace('_', ' ').title()[:30] for r in top_reasons.keys()], fontsize=9)
            ax4.set_xlabel('Count', fontsize=12, fontweight='bold')
            ax4.set_title('Top Decision Reasons', fontsize=13, fontweight='bold', pad=10)
            ax4.grid(axis='x', alpha=0.3, linestyle='--')
            
            for i, (bar, val) in enumerate(zip(bars4, top_reasons.values())):
                width = bar.get_width()
                pct = val/total*100
                ax4.text(width + max(top_reasons.values())*0.01, bar.get_y() + bar.get_height()/2,
                        f'{int(val)} ({pct:.1f}%)', va='center', fontsize=9, fontweight='bold')
        
        # 5. Net Impact Summary (Middle Right)
        ax5 = fig.add_subplot(gs[1, 1:])
        net_improvement = improvements['intent_improved'] - improvements['intent_degraded']
        net_improvement_pct = (improvements['intent_improved'] - improvements['intent_degraded']) / total * 100
        acc_improvement = accuracy['hybrid_intent_accuracy'] - accuracy['raw_intent_accuracy']
        
        summary_text = f"""
HYBRID SYSTEM INTERVENTION SUMMARY

Total Test Cases: {total}

INTERVENTION LEVEL:
  • Intent Changed: {summary['intent_changed']} ({summary['intent_changed_percentage']:.1f}%)
  • Command Changed: {summary['command_changed']} ({summary['command_changed_percentage']:.1f}%)
  • Confidence Changed: {summary['confidence_changed']} ({summary['confidence_changed_percentage']:.1f}%)
  • Entities Changed: {summary['entities_changed']} ({summary['entities_changed_percentage']:.1f}%)

QUALITY IMPACT:
  • Intent Improved: {improvements['intent_improved']} ({improvements['intent_improved_percentage']:.1f}%)
  • Intent Degraded: {improvements['intent_degraded']} ({improvements['intent_degraded_percentage']:.1f}%)
  • Net Improvement: {net_improvement} cases ({net_improvement_pct:+.1f}%)

ACCURACY:
  • Raw Model: {accuracy['raw_intent_accuracy']:.1%}
  • Hybrid System: {accuracy['hybrid_intent_accuracy']:.1%}
  • Improvement: {acc_improvement:+.1%}

CONCLUSION:
  Hybrid System shows {'POSITIVE' if net_improvement > 0 else 'NEGATIVE'} impact
  with {'LOW' if summary['intent_changed_percentage'] < 10 else 'MODERATE' if summary['intent_changed_percentage'] < 20 else 'HIGH'} intervention level
        """
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax5.axis('off')
        ax5.set_title('Executive Summary', fontsize=14, fontweight='bold', pad=15)
        
        # 6. Intervention Level Gauge (Bottom)
        ax6 = fig.add_subplot(gs[2, 1:])
        self._create_intervention_gauge(ax6, summary['intent_changed_percentage'])
        
        plt.suptitle('Hybrid System Intervention Analysis Dashboard', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(output_dir / 'hybrid_intervention_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Main dashboard: {output_dir / 'hybrid_intervention_dashboard.png'}")
    
    def _create_intervention_gauge(self, ax, intervention_pct: float):
        """Tạo gauge chart cho intervention level"""
        # Gauge chart
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Color zones
        low_zone = (0, 10)
        medium_zone = (10, 20)
        high_zone = (20, 100)
        
        # Draw zones
        if intervention_pct <= low_zone[1]:
            color = '#2ecc71'  # Green
            zone = 'LOW'
        elif intervention_pct <= medium_zone[1]:
            color = '#f39c12'  # Orange
            zone = 'MODERATE'
        else:
            color = '#e74c3c'  # Red
            zone = 'HIGH'
        
        # Draw gauge
        ax.plot(theta, r, 'k-', linewidth=3)
        ax.fill_between(theta, 0, r, alpha=0.2, color=color)
        
        # Draw needle
        needle_angle = np.pi * (1 - intervention_pct / 100)
        ax.plot([needle_angle, needle_angle], [0, 1.1], 'r-', linewidth=3, label=f'{intervention_pct:.1f}%')
        
        # Labels
        ax.text(0, 1.3, f'Intervention Level: {zone}', ha='center', fontsize=14, fontweight='bold', color=color)
        ax.text(0, 1.15, f'{intervention_pct:.2f}%', ha='center', fontsize=16, fontweight='bold')
        ax.text(np.pi/2, 0.3, '0%', ha='center', fontsize=10)
        ax.text(0, 0.3, '50%', ha='center', fontsize=10)
        ax.text(-np.pi/2, 0.3, '100%', ha='center', fontsize=10)
        
        ax.set_ylim([0, 1.5])
        ax.set_xlim([-np.pi/2 - 0.2, np.pi/2 + 0.2])
        ax.axis('off')
        ax.set_title('Intervention Level Gauge', fontsize=13, fontweight='bold', pad=10)
    
    def _create_intervention_level_chart(self, output_dir: Path):
        """Tạo chart chi tiết về intervention level"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        summary = self.report_data['summary']
        total = self.report_data['summary']['total_samples']
        
        categories = ['Intent', 'Command', 'Confidence', 'Entities']
        changed = [
            summary['intent_changed'],
            summary['command_changed'],
            summary['confidence_changed'],
            summary['entities_changed']
        ]
        unchanged = [total - c for c in changed]
        percentages = [c/total*100 for c in changed]
        
        x = np.arange(len(categories))
        width = 0.6
        
        bars1 = ax.bar(x, unchanged, width, label='No Change', color='#95a5a6', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x, changed, width, bottom=unchanged, label='Changed', color='#e74c3c', alpha=0.7, edgecolor='black')
        
        # Add percentage labels
        for i, (c, pct) in enumerate(zip(changed, percentages)):
            if c > 0:
                ax.text(i, total - c/2, f'{pct:.1f}%', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white')
        
        ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        ax.set_xlabel('Intervention Type', fontsize=12, fontweight='bold')
        ax.set_title('Intervention Level by Type', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'intervention_level_detail.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Intervention level chart: {output_dir / 'intervention_level_detail.png'}")
    
    def _create_improvement_chart(self, output_dir: Path):
        """Tạo chart về improvement vs degradation"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        improvements = self.report_data['improvements']
        total = self.report_data['summary']['total_samples']
        
        # Pie chart for overall impact
        labels = ['Improved', 'Degraded', 'No Change']
        sizes = [
            improvements['intent_improved'] + improvements['command_improved'],
            improvements['intent_degraded'] + improvements['command_degraded'],
            total - (improvements['intent_improved'] + improvements['command_improved'] + 
                    improvements['intent_degraded'] + improvements['command_degraded'])
        ]
        colors_pie = ['#2ecc71', '#e74c3c', '#95a5a6']
        explode = (0.1, 0.1, 0)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('Overall Impact Distribution', fontsize=13, fontweight='bold', pad=15)
        
        # Bar chart for detailed breakdown
        categories = ['Intent\nImproved', 'Intent\nDegraded', 'Command\nImproved', 'Command\nDegraded']
        values = [
            improvements['intent_improved'],
            improvements['intent_degraded'],
            improvements['command_improved'],
            improvements['command_degraded']
        ]
        colors_bar = ['#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c']
        
        bars = ax2.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        ax2.set_title('Improvement vs Degradation Breakdown', fontsize=13, fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            pct = val/total*100 if total > 0 else 0
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    f'{int(val)}\n({pct:.1f}%)', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Improvement chart: {output_dir / 'improvement_analysis.png'}")
    
    def _create_accuracy_comparison(self, output_dir: Path):
        """Tạo chart so sánh accuracy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        accuracy = self.report_data['accuracy']
        
        # Intent accuracy comparison
        intent_data = {
            'Raw Model': accuracy['raw_intent_accuracy'],
            'Hybrid System': accuracy['hybrid_intent_accuracy']
        }
        colors_intent = ['#95a5a6', '#3498db']
        bars1 = ax1.bar(list(intent_data.keys()), list(intent_data.values()),
                       color=colors_intent, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Intent Classification Accuracy', fontsize=13, fontweight='bold', pad=15)
        ax1.set_ylim([0, 1.1])
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        diff_intent = accuracy['hybrid_intent_accuracy'] - accuracy['raw_intent_accuracy']
        ax1.text(0.5, 1.05, f'Difference: {diff_intent:+.3f} ({diff_intent*100:+.2f}%)',
                ha='center', transform=ax1.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        for bar, val in zip(bars1, intent_data.values()):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Command accuracy comparison
        command_data = {
            'Raw Model': accuracy['raw_command_accuracy'],
            'Hybrid System': accuracy['hybrid_command_accuracy']
        }
        colors_cmd = ['#95a5a6', '#3498db']
        bars2 = ax2.bar(list(command_data.keys()), list(command_data.values()),
                       color=colors_cmd, alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Command Classification Accuracy', fontsize=13, fontweight='bold', pad=15)
        ax2.set_ylim([0, 1.1])
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        diff_cmd = accuracy['hybrid_command_accuracy'] - accuracy['raw_command_accuracy']
        ax2.text(0.5, 1.05, f'Difference: {diff_cmd:+.3f} ({diff_cmd*100:+.2f}%)',
                ha='center', transform=ax2.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        for bar, val in zip(bars2, command_data.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Accuracy comparison: {output_dir / 'accuracy_comparison.png'}")
    
    def _create_decision_reasons_chart(self, output_dir: Path):
        """Tạo chart về decision reasons"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        decision_reasons = self.report_data['decision_reasons']
        total = self.report_data['summary']['total_samples']
        
        # Sort by count
        sorted_reasons = sorted(decision_reasons.items(), key=lambda x: x[1], reverse=True)
        reasons = [r[0].replace('_', ' ').title() for r, _ in sorted_reasons]
        counts = [c for _, c in sorted_reasons]
        percentages = [c/total*100 for c in counts]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(reasons))
        bars = ax.barh(y_pos, counts, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(reasons, fontsize=10)
        ax.set_xlabel('Number of Cases', fontsize=12, fontweight='bold')
        ax.set_title('Decision Reason Distribution', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
            width = bar.get_width()
            ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{int(count)} ({pct:.1f}%)', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'decision_reasons.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Decision reasons chart: {output_dir / 'decision_reasons.png'}")
    
    def _create_confidence_analysis(self, output_dir: Path):
        """Tạo chart phân tích confidence changes"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Load comparison results if available
        comparison_file = self.report_path.parent / 'comparison_results_sample.json'
        if comparison_file.exists():
            with open(comparison_file, 'r', encoding='utf-8') as f:
                comparison_data = json.load(f)
            
            confidence_diffs = [
                r['hybrid_confidence'] - r['raw_confidence']
                for r in comparison_data
                if 'hybrid_confidence' in r and 'raw_confidence' in r
            ]
            
            if confidence_diffs:
                # Histogram
                ax1.hist(confidence_diffs, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
                ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
                ax1.set_xlabel('Confidence Difference (Hybrid - Raw)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax1.set_title('Confidence Change Distribution', fontsize=13, fontweight='bold', pad=15)
                ax1.legend()
                ax1.grid(alpha=0.3, linestyle='--')
                
                # Box plot
                bp = ax2.boxplot([confidence_diffs], vert=True, patch_artist=True,
                                tick_labels=['Confidence\nDifference'])
                bp['boxes'][0].set_facecolor('#3498db')
                bp['boxes'][0].set_alpha(0.7)
                ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
                ax2.set_ylabel('Confidence Difference', fontsize=12, fontweight='bold')
                ax2.set_title('Confidence Change Statistics', fontsize=13, fontweight='bold', pad=15)
                ax2.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Add statistics
                mean_diff = np.mean(confidence_diffs)
                median_diff = np.median(confidence_diffs)
                stats_text = f'Mean: {mean_diff:.3f}\nMedian: {median_diff:.3f}'
                ax2.text(0.5, 0.95, stats_text, transform=ax2.transAxes,
                        ha='center', va='top', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            # Use summary statistics if available
            conf_stats = self.report_data.get('confidence_statistics', {})
            if conf_stats:
                ax1.text(0.5, 0.5, f"Mean: {conf_stats.get('mean_diff', 0):.3f}\n"
                                  f"Std: {conf_stats.get('std_diff', 0):.3f}\n"
                                  f"Min: {conf_stats.get('min_diff', 0):.3f}\n"
                                  f"Max: {conf_stats.get('max_diff', 0):.3f}",
                        ha='center', va='center', transform=ax1.transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax1.set_title('Confidence Statistics', fontsize=13, fontweight='bold')
                ax1.axis('off')
                ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Confidence analysis: {output_dir / 'confidence_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Hybrid System Intervention Analysis')
    parser.add_argument('--report-path', type=str, 
                       default='reports/hybrid_intervention_test/hybrid_intervention_report.json',
                       help='Path to intervention report JSON file')
    parser.add_argument('--output-dir', type=str, default='reports/hybrid_intervention_visualization',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    report_path = Path(args.report_path)
    if not report_path.exists():
        print(f"[ERROR] Report file not found: {report_path}")
        return
    
    visualizer = HybridInterventionVisualizer(report_path)
    visualizer.create_comprehensive_dashboard(Path(args.output_dir))
    
    print("\n" + "="*80)
    print("[OK] All visualizations created successfully!")
    print("="*80)


if __name__ == '__main__':
    main()

