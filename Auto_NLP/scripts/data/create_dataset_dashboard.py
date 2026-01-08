#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tạo dashboard HTML trực quan hóa dataset với bố cục đẹp và dễ nhìn.
Kết hợp tất cả visualizations và thống kê vào một trang web.

Usage:
    python scripts/data/create_dataset_dashboard.py
    python scripts/data/create_dataset_dashboard.py --output reports/dataset_dashboard.html
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 100

def load_json(path: Path) -> List[Dict]:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def create_visualizations(
    master_data: List[Dict],
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    output_dir: Path,
) -> Dict[str, str]:
    """Tạo tất cả visualizations và trả về paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = {}
    
    print("[INFO] Dang tao visualizations...")
    
    # 1. Intent Distribution - Bar Chart
    print("  - Intent distribution bar chart...")
    all_data = master_data
    intent_counter = Counter(s.get("intent", s.get("command", "unknown")) for s in all_data)
    intents = sorted(intent_counter.keys())
    counts = [intent_counter[i] for i in intents]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(intents)), counts, color=plt.cm.Set3(np.linspace(0, 1, len(intents))))
    ax.set_xticks(range(len(intents)))
    ax.set_xticklabels(intents, rotation=45, ha='right')
    ax.set_ylabel('Số lượng mẫu', fontsize=11)
    ax.set_title('Phân bố Intent trong Dataset', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    path = output_dir / "dashboard_intent_bar.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    image_paths['intent_bar'] = str(path.relative_to(output_dir.parent))
    
    # 2. Intent Distribution - Pie Chart
    print("  - Intent distribution pie chart...")
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, len(intents)))
    wedges, texts, autotexts = ax.pie(counts, labels=intents, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax.set_title('Phân bố Intent (Pie Chart)', fontsize=14, fontweight='bold', pad=20)
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    plt.tight_layout()
    path = output_dir / "dashboard_intent_pie.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    image_paths['intent_pie'] = str(path.relative_to(output_dir.parent))
    
    # 3. Train/Val/Test Split
    print("  - Train/Val/Test split...")
    split_counts = [len(train_data), len(val_data), len(test_data)]
    split_labels = ['Train', 'Validation', 'Test']
    split_colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    bars = ax1.bar(split_labels, split_counts, color=split_colors)
    ax1.set_ylabel('Số lượng mẫu', fontsize=11)
    ax1.set_title('Phân chia Train/Val/Test', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    for bar, count in zip(bars, split_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart
    total = sum(split_counts)
    percentages = [c/total*100 for c in split_counts]
    ax2.pie(split_counts, labels=[f'{l}\n({c:,})\n({p:.1f}%)' 
                                   for l, c, p in zip(split_labels, split_counts, percentages)],
            colors=split_colors, autopct='', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Tỷ lệ phân chia', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    path = output_dir / "dashboard_split.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    image_paths['split'] = str(path.relative_to(output_dir.parent))
    
    # 4. Entity Distribution
    print("  - Entity distribution...")
    entity_counter = Counter()
    for sample in all_data:
        for ent in sample.get("entities", []):
            if isinstance(ent, dict):
                label = ent.get("label", ent.get("type", "UNKNOWN"))
                entity_counter[label] += 1
    
    top_entities = entity_counter.most_common(15)
    entity_labels = [e[0] for e in top_entities]
    entity_counts = [e[1] for e in top_entities]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(range(len(entity_labels)), entity_counts, color='#55A868')
    ax.set_xticks(range(len(entity_labels)))
    ax.set_xticklabels(entity_labels, rotation=45, ha='right')
    ax.set_ylabel('Số lần xuất hiện', fontsize=11)
    ax.set_title('Top 15 Entity Types', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar, count in zip(bars, entity_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    path = output_dir / "dashboard_entity_dist.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    image_paths['entity_dist'] = str(path.relative_to(output_dir.parent))
    
    # 5. Sentence Length Distribution
    print("  - Sentence length distribution...")
    all_lengths = []
    for sample in all_data:
        text = sample.get("input", "")
        words = text.split()
        all_lengths.append(len(words))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(all_lengths, bins=30, edgecolor='black', alpha=0.7, color='#8172B3')
    ax.set_xlabel('Độ dài câu (số từ)', fontsize=11)
    ax.set_ylabel('Số lượng mẫu', fontsize=11)
    ax.set_title('Phân bố độ dài câu', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    mean_len = np.mean(all_lengths)
    median_len = np.median(all_lengths)
    ax.axvline(mean_len, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_len:.1f}')
    ax.axvline(median_len, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_len:.1f}')
    ax.legend()
    
    plt.tight_layout()
    path = output_dir / "dashboard_sentence_length.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    image_paths['sentence_length'] = str(path.relative_to(output_dir.parent))
    
    # 6. Entities per Sample
    print("  - Entities per sample...")
    entity_counts_per_sample = [len(s.get("entities", [])) for s in all_data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = range(0, min(max(entity_counts_per_sample) + 2, 15))
    ax.hist(entity_counts_per_sample, bins=bins, edgecolor='black', alpha=0.7, color='#F18F01')
    ax.set_xlabel('Số entity trong một mẫu', fontsize=11)
    ax.set_ylabel('Số lượng mẫu', fontsize=11)
    ax.set_title('Phân bố số entity/mẫu', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    mean_ent = np.mean(entity_counts_per_sample)
    ax.axvline(mean_ent, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ent:.2f}')
    ax.legend()
    
    plt.tight_layout()
    path = output_dir / "dashboard_entities_per_sample.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    image_paths['entities_per_sample'] = str(path.relative_to(output_dir.parent))
    
    # 7. Entity by Intent Heatmap
    print("  - Entity by intent heatmap...")
    entities_by_intent = defaultdict(Counter)
    for sample in all_data:
        intent = sample.get("intent", sample.get("command", "unknown"))
        for ent in sample.get("entities", []):
            if isinstance(ent, dict):
                label = ent.get("label", ent.get("type", "UNKNOWN"))
                entities_by_intent[intent][label] += 1
    
    # Prepare heatmap data
    all_intents = sorted(set(s.get("intent", s.get("command", "unknown")) for s in all_data))
    top_entity_types = [e[0] for e in entity_counter.most_common(12)]
    
    heatmap_data = []
    for intent in all_intents:
        row = [entities_by_intent[intent].get(et, 0) for et in top_entity_types]
        heatmap_data.append(row)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(heatmap_data, 
                xticklabels=top_entity_types,
                yticklabels=all_intents,
                annot=True, fmt='d', cmap='YlOrRd',
                cbar_kws={'label': 'Số lần xuất hiện'})
    ax.set_xlabel('Entity Types', fontsize=11)
    ax.set_ylabel('Intents', fontsize=11)
    ax.set_title('Entity Types theo Intent (Heatmap)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = output_dir / "dashboard_entity_intent_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    image_paths['entity_intent_heatmap'] = str(path.relative_to(output_dir.parent))
    
    print("[OK] Da tao tat ca visualizations!")
    return image_paths

def generate_html_dashboard(
    aggregate_report: Dict[str, Any],
    image_paths: Dict[str, str],
    output_path: Path,
) -> None:
    """Tạo HTML dashboard."""
    master = aggregate_report.get("master", {})
    summary = aggregate_report.get("summary", {})
    consistency = aggregate_report.get("consistency_check", {})
    
    # Get statistics
    stats = master.get("statistics", {})
    intent_dist = master.get("intent_distribution", {})
    entity_dist = master.get("entity_type_distribution", {})
    
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Dashboard - Auto NLP</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            font-size: 2em;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .stat-card p {{
            color: #666;
            font-size: 0.9em;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .image-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
        }}
        .image-card h4 {{
            margin-top: 10px;
            color: #667eea;
            text-align: center;
        }}
        .table-container {{
            overflow-x: auto;
            margin-bottom: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .badge-success {{
            background: #28a745;
            color: white;
        }}
        .badge-warning {{
            background: #ffc107;
            color: #333;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Dataset Dashboard</h1>
            <p>Auto NLP System - Dataset Analysis Report</p>
            <p style="margin-top: 10px; font-size: 0.9em;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <!-- Overview Statistics -->
            <div class="section">
                <h2 class="section-title">📈 Tổng quan</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>{summary.get('total_samples', {}).get('master', 0):,}</h3>
                        <p>Total Samples (Master)</p>
                    </div>
                    <div class="stat-card">
                        <h3>{summary.get('total_samples', {}).get('train', 0):,}</h3>
                        <p>Train Samples</p>
                    </div>
                    <div class="stat-card">
                        <h3>{summary.get('total_samples', {}).get('val', 0):,}</h3>
                        <p>Validation Samples</p>
                    </div>
                    <div class="stat-card">
                        <h3>{summary.get('total_samples', {}).get('test', 0):,}</h3>
                        <p>Test Samples</p>
                    </div>
                    <div class="stat-card">
                        <h3>{summary.get('unique_intents', 0)}</h3>
                        <p>Unique Intents</p>
                    </div>
                    <div class="stat-card">
                        <h3>{summary.get('unique_entity_types', 0)}</h3>
                        <p>Entity Types</p>
                    </div>
                    <div class="stat-card">
                        <h3>{stats.get('avg_entities_per_sample', 0):.2f}</h3>
                        <p>Avg Entities/Sample</p>
                    </div>
                    <div class="stat-card">
                        <h3>{stats.get('avg_sentence_length', 0):.1f}</h3>
                        <p>Avg Sentence Length</p>
                    </div>
                </div>
                
                <div class="table-container">
                    <h3 style="margin-bottom: 15px; color: #667eea;">Consistency Check</h3>
                    <table>
                        <tr>
                            <th>Check</th>
                            <th>Value</th>
                            <th>Status</th>
                        </tr>
                        <tr>
                            <td>Master = Train+Val+Test</td>
                            <td>{consistency.get('master_count', 0):,} = {consistency.get('split_total', 0):,}</td>
                            <td>
                                <span class="badge {'badge-success' if consistency.get('matches', False) else 'badge-warning'}">
                                    {'✅ Match' if consistency.get('matches', False) else '⚠️ Mismatch'}
                                </span>
                            </td>
                        </tr>
                        <tr>
                            <td>Train Ratio</td>
                            <td>{consistency.get('train_ratio', 0):.2f}%</td>
                            <td><span class="badge badge-success">✅ OK</span></td>
                        </tr>
                        <tr>
                            <td>Val Ratio</td>
                            <td>{consistency.get('val_ratio', 0):.2f}%</td>
                            <td><span class="badge badge-success">✅ OK</span></td>
                        </tr>
                        <tr>
                            <td>Test Ratio</td>
                            <td>{consistency.get('test_ratio', 0):.2f}%</td>
                            <td><span class="badge badge-success">✅ OK</span></td>
                        </tr>
                    </table>
                </div>
            </div>
            
            <!-- Visualizations -->
            <div class="section">
                <h2 class="section-title">📊 Visualizations</h2>
                <div class="image-grid">
                    <div class="image-card">
                        <img src="{image_paths.get('intent_bar', '')}" alt="Intent Distribution">
                        <h4>Intent Distribution (Bar Chart)</h4>
                    </div>
                    <div class="image-card">
                        <img src="{image_paths.get('intent_pie', '')}" alt="Intent Pie Chart">
                        <h4>Intent Distribution (Pie Chart)</h4>
                    </div>
                    <div class="image-card">
                        <img src="{image_paths.get('split', '')}" alt="Train/Val/Test Split">
                        <h4>Train/Val/Test Split</h4>
                    </div>
                    <div class="image-card">
                        <img src="{image_paths.get('entity_dist', '')}" alt="Entity Distribution">
                        <h4>Entity Type Distribution</h4>
                    </div>
                    <div class="image-card">
                        <img src="{image_paths.get('sentence_length', '')}" alt="Sentence Length">
                        <h4>Sentence Length Distribution</h4>
                    </div>
                    <div class="image-card">
                        <img src="{image_paths.get('entities_per_sample', '')}" alt="Entities per Sample">
                        <h4>Entities per Sample</h4>
                    </div>
                    <div class="image-card" style="grid-column: 1 / -1;">
                        <img src="{image_paths.get('entity_intent_heatmap', '')}" alt="Entity by Intent Heatmap">
                        <h4>Entity Types by Intent (Heatmap)</h4>
                    </div>
                </div>
            </div>
            
            <!-- Intent Distribution Table -->
            <div class="section">
                <h2 class="section-title">🎯 Intent Distribution</h2>
                <div class="table-container">
                    <table>
                        <tr>
                            <th>Intent</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
"""
    
    # Add intent distribution rows
    total = master.get("total_samples", 1)
    for intent, count in sorted(intent_dist.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        html_content += f"""
                        <tr>
                            <td><strong>{intent}</strong></td>
                            <td>{count:,}</td>
                            <td>{pct:.2f}%</td>
                        </tr>
"""
    
    html_content += f"""
                    </table>
                </div>
            </div>
            
            <!-- Entity Distribution Table -->
            <div class="section">
                <h2 class="section-title">🏷️ Top Entity Types</h2>
                <div class="table-container">
                    <table>
                        <tr>
                            <th>Entity Type</th>
                            <th>Count</th>
                        </tr>
"""
    
    # Add entity distribution rows
    for entity, count in list(entity_dist.items())[:20]:
        html_content += f"""
                        <tr>
                            <td><strong>{entity}</strong></td>
                            <td>{count:,}</td>
                        </tr>
"""
    
    html_content += """
                    </table>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Auto NLP System - Dataset Dashboard</p>
            <p style="font-size: 0.85em; margin-top: 5px;">Generated automatically from dataset analysis</p>
        </div>
    </div>
</body>
</html>
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"[OK] Da tao HTML dashboard: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Tạo dashboard HTML trực quan hóa dataset.")
    parser.add_argument(
        "--master",
        type=Path,
        default=Path("src/data/raw/elderly_commands_master.json"),
        help="Đường dẫn master dataset.",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("src/data/processed/train.json"),
        help="Đường dẫn train dataset.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path("src/data/processed/val.json"),
        help="Đường dẫn val dataset.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("src/data/processed/test.json"),
        help="Đường dẫn test dataset.",
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=Path("reports/dataset_aggregate_report.json"),
        help="Đường dẫn aggregate report JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dataset_dashboard.html"),
        help="Đường dẫn file output HTML.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("reports"),
        help="Thư mục lưu images.",
    )
    args = parser.parse_args()
    
    # Load data
    print("[INFO] Dang load datasets...")
    master_data = load_json(args.master)
    train_data = load_json(args.train)
    val_data = load_json(args.val)
    test_data = load_json(args.test)
    
    # Load or create aggregate report
    if args.aggregate_report.exists():
        print(f"[INFO] Dang load aggregate report: {args.aggregate_report}")
        with args.aggregate_report.open("r", encoding="utf-8") as f:
            aggregate_report = json.load(f)
    else:
        print("[WARN] Aggregate report khong ton tai, tao moi...")
        from aggregate_datasets import aggregate_datasets
        aggregate_report = aggregate_datasets(args.master, args.train, args.val, args.test)
    
    # Create visualizations
    image_paths = create_visualizations(
        master_data, train_data, val_data, test_data, args.image_dir
    )
    
    # Generate HTML dashboard
    generate_html_dashboard(aggregate_report, image_paths, args.output)
    
    print(f"\n{'='*70}")
    print("[OK] Hoan tat!")
    print(f"Dashboard HTML: {args.output}")
    print(f"Images directory: {args.image_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

