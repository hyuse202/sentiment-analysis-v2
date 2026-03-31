#!/usr/bin/env python3
"""
Generate figures for final report
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Set up Vietnamese font support
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 150

output_dir = "docs/final_report/figures"
os.makedirs(output_dir, exist_ok=True)

# ========== 1. Dataset Class Distribution ==========
print("Generating class distribution chart...")

labels = ['POSITIVE\n(480 samples)', 'NEGATIVE\n(258 samples)', 'NEUTRAL\n(159 samples)']
sizes = [480, 258, 159]
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
explode = (0.05, 0, 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.set_title('Phân phối các lớp Sentiment (897 samples)', fontsize=12, fontweight='bold')

# Bar chart
sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
counts = [480, 258, 159]
bars = ax2.bar(sentiments, counts, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_xlabel('Sentiment Class', fontsize=11)
ax2.set_ylabel('Số mẫu', fontsize=11)
ax2.set_title('Số lượng mẫu theo lớp', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 550)

# Add value labels on bars
for bar, count in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{output_dir}/01_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/01_class_distribution.png")

# ========== 2. 3-Class vs Binary Comparison ==========
print("Generating 3-class vs binary comparison chart...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 3-class results
models_3class = ['SVM (RBF)', 'Random Forest', 'Logistic Reg', 'SVM (Linear)', 'Naive Bayes']
acc_3class = [66.7, 65.9, 65.9, 62.2, 54.1]
colors_3class = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#e74c3c']

bars1 = ax1.barh(models_3class, acc_3class, color=colors_3class, edgecolor='black')
ax1.set_xlabel('Accuracy (%)', fontsize=11)
ax1.set_title('Kết quả 3-class Classification\n(Best: 66.7%)', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 100)
ax1.axvline(x=80, color='red', linestyle='--', label='Target 80%')

for bar, acc in zip(bars1, acc_3class):
    ax1.text(acc + 1, bar.get_y() + bar.get_height()/2, f'{acc:.1f}%',
             va='center', fontsize=10, fontweight='bold')

# Binary results progression
stages = ['Baseline\n(Balanced)', 'Binary\nClassification', 'Optimized', 'Stacking', 'Final\nBest']
acc_binary = [65.9, 78.5, 79.3, 78.9, 84.4]
colors_binary = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#2ecc71']

bars2 = ax2.bar(stages, acc_binary, color=colors_binary, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Giai đoạn cải tiến', fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_title('Tiến trình cải thiện Accuracy (Binary)\n(Target: 80%)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Target 80%')

for bar, acc in zip(bars2, acc_binary):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1f}%',
             ha='center', fontsize=10, fontweight='bold')

ax2.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f"{output_dir}/02_accuracy_progression.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/02_accuracy_progression.png")

# ========== 3. Model Comparison (Unique Results Only) ==========
print("Generating model comparison chart...")

# Collect all unique results
all_results = {
    # 3-class models
    'Naive Bayes (3-class)': 54.1,
    'Logistic Regression (3-class)': 65.9,
    'SVM Linear (3-class)': 62.2,
    'SVM RBF (3-class)': 66.7,
    'Random Forest (3-class)': 65.9,
    # Binary models (unique best results)
    'XGBoost (Binary)': 78.5,
    'SVM RBF (Binary)': 75.6,
    'Logistic Regression (Binary)': 77.8,
    'Random Forest (Binary)': 77.0,
    'Ensemble (Binary)': 79.3,
    'Stacking (Binary)': 78.9,
    'XGBoost (Final)': 84.4,
}

# Group for visualization
fig, ax = plt.subplots(figsize=(14, 8))

models = list(all_results.keys())
accuracies = list(all_results.values())

# Color based on type
colors = []
for m in models:
    if '3-class' in m:
        colors.append('#e74c3c')  # Red for 3-class
    elif 'Final' in m:
        colors.append('#2ecc71')  # Green for final best
    else:
        colors.append('#3498db')  # Blue for binary

y_pos = np.arange(len(models))
bars = ax.barh(y_pos, accuracies, color=colors, edgecolor='black', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=10)
ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_title('So sánh Accuracy các mô hình (Loại bỏ kết quả trùng)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
ax.axvline(x=80, color='red', linestyle='--', linewidth=2, label='Target 80%')

# Add value labels
for bar, acc in zip(bars, accuracies):
    ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2, f'{acc:.1f}%',
            va='center', fontsize=9, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', edgecolor='black', label='3-class Classification'),
    Patch(facecolor='#3498db', edgecolor='black', label='Binary Classification'),
    Patch(facecolor='#2ecc71', edgecolor='black', label='Best Model (Final)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(f"{output_dir}/03_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/03_model_comparison.png")

# ========== 4. Confusion Matrix for Best Model ==========
print("Generating confusion matrix for best model...")

fig, ax = plt.subplots(figsize=(8, 6))

# Best model confusion matrix (from results)
cm = np.array([[18, 3],
               [4, 20]])

im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)

classes = ['NON-POSITIVE', 'POSITIVE']
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       xlabel='Predicted Label', ylabel='True Label',
       title='Confusion Matrix - Best Model (XGBoost, 84.4%)')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{output_dir}/04_confusion_matrix_best.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/04_confusion_matrix_best.png")

# ========== 5. Feature Engineering Pipeline ==========
print("Generating pipeline diagram...")

fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')

# Pipeline stages
stages = [
    ('Raw Data\n(897 samples)', 1, '#e74c3c'),
    ('Preprocessing\n(Lowercase, Remove\nStopwords)', 3.5, '#f39c12'),
    ('TF-IDF\n(8000 features,\n1-4 grams)', 6, '#3498db'),
    ('XGBoost\nClassifier', 8.5, '#9b59b6'),
    ('Prediction\n(84.4% acc)', 11, '#2ecc71'),
]

for text, x, color in stages:
    rect = plt.Rectangle((x, 1), 2, 2, facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x + 1, 2, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Arrows
arrow_style = dict(arrowstyle='->', lw=2, color='black')
ax.annotate('', xy=(3.4, 2), xytext=(3.1, 2), arrowprops=arrow_style)
ax.annotate('', xy=(5.9, 2), xytext=(5.6, 2), arrowprops=arrow_style)
ax.annotate('', xy=(8.4, 2), xytext=(8.1, 2), arrowprops=arrow_style)
ax.annotate('', xy=(10.9, 2), xytext=(10.6, 2), arrowprops=arrow_style)

ax.set_title('Machine Learning Pipeline', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f"{output_dir}/05_pipeline.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/05_pipeline.png")

# ========== 6. Key Findings Summary ==========
print("Generating key findings summary...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

findings = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        KẾT QUẢ CHÍNH (KEY FINDINGS)                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ✅ BEST MODEL: XGBoost                                                      ║
║     • Accuracy: 84.4% (TARGET >= 80% ACHIEVED)                               ║
║     • Test size: 5% (45 samples)                                             ║
║     • Features: 8000 TF-IDF (1-4 grams)                                      ║
║                                                                              ║
║  📊 3-CLASS vs BINARY:                                                       ║
║     • 3-class best: 66.7% (SVM RBF)                                          ║
║     • Binary best: 84.4% (XGBoost)                                           ║
║     → Binary classification hiệu quả hơn ~18%                                ║
║                                                                              ║
║  🔧 CÁC TECHNIQUE HIỆU QUẢ:                                                  ║
║     • Loại bỏ Vietnamese stopwords (+3-5%)                                   ║
║     • TF-IDF n-grams (1-4) (+2-3%)                                           ║
║     • Class weight balancing (+5-8%)                                         ║
║     • Tăng training data (95% train) (+5-6%)                                 ║
║                                                                              ║
║  ⚠️ CÁC TECHNIQUE KHÔNG HIỆU QUẢ:                                            ║
║     • SMOTE → Giảm accuracy (từ 79.3% → 77.8%)                               ║
║     • Stacking Ensemble → Không cải thiện đáng kể                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

ax.text(0.5, 0.5, findings, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

plt.savefig(f"{output_dir}/06_key_findings.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/06_key_findings.png")

print("\n✅ All figures generated successfully!")
print(f"Output directory: {output_dir}")
