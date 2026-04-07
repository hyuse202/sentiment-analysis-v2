#!/usr/bin/env python3
"""
Generate figures for final report
Updated for 5,322 samples dataset with 77.4% accuracy
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

labels = ['POSITIVE\n(2,866 samples)', 'NEGATIVE\n(1,715 samples)', 'NEUTRAL\n(741 samples)']
sizes = [2866, 1715, 741]
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
explode = (0.05, 0, 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.set_title('Phân phối các lớp Sentiment (5,322 samples)', fontsize=12, fontweight='bold')

# Bar chart
sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
counts = [2866, 1715, 741]
bars = ax2.bar(sentiments, counts, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_xlabel('Sentiment Class', fontsize=11)
ax2.set_ylabel('Số mẫu', fontsize=11)
ax2.set_title('Số lượng mẫu theo lớp', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 3500)

# Add value labels on bars
for bar, count in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{output_dir}/01_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/01_class_distribution.png")

# ========== 2. Test Size Impact on Accuracy ==========
print("Generating test size impact chart...")

fig, ax = plt.subplots(figsize=(12, 6))

test_sizes = ['3%\n(160 samples)', '5%\n(267 samples)', '10%\n(533 samples)', '20%\n(1,065 samples)']
accuracies = [81.9, 79.8, 76.9, 77.4]
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

bars = ax.bar(test_sizes, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Test Size', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Tác động của Test Size đến Accuracy\n(Larger test size = more reliable evaluation)', fontsize=14, fontweight='bold')
ax.set_ylim(70, 85)
ax.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Target 80%')

# Add value labels
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')

ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f"{output_dir}/02_accuracy_progression.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/02_accuracy_progression.png")

# ========== 3. Model Comparison (20% Test Size) ==========
print("Generating model comparison chart...")

models = ['Ensemble\n(XGB+SVM+LR)', 'Logistic\nRegression', 'SVM\n(RBF)', 'XGBoost', 'Random\nForest']
accuracies = [77.4, 77.0, 76.8, 76.4, 74.9]
colors = ['#2ecc71', '#3498db', '#3498db', '#3498db', '#e74c3c']

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.barh(models, accuracies, color=colors, edgecolor='black', linewidth=1)
ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_title('So sánh Accuracy các mô hình (20% Test Size)', fontsize=14, fontweight='bold')
ax.set_xlim(70, 80)
ax.axvline(x=80, color='red', linestyle='--', linewidth=2, label='Target 80%')

# Add value labels
for bar, acc in zip(bars, accuracies):
    ax.text(acc + 0.2, bar.get_y() + bar.get_height()/2,
             f'{acc:.1f}%', va='center', fontsize=11, fontweight='bold')

ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f"{output_dir}/03_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/03_model_comparison.png")

# ========== 4. Confusion Matrix for Best Model ==========
print("Generating confusion matrix for best model...")

fig, ax = plt.subplots(figsize=(8, 6))

# Best model confusion matrix (Ensemble, 20% test)
cm = np.array([[394, 97],
               [144, 430]])

im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)

classes = ['NON-POSITIVE', 'POSITIVE']
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       xlabel='Predicted Label', ylabel='True Label',
       title='Confusion Matrix - Ensemble (77.4% on 20% test)')

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
    ('Raw Data\n(5,322 samples)', 1, '#e74c3c'),
    ('Preprocessing\n(Lowercase, Remove\nStopwords)', 3.5, '#f39c12'),
    ('TF-IDF\n(10,000 features,\n1-5 grams)', 6, '#3498db'),
    ('Ensemble\n(XGB+SVM+LR)', 8.5, '#9b59b6'),
    ('Prediction\n(77.4% accuracy)', 11, '#2ecc71'),
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
║                        KẾt QUẢ CHÍNH (KEY FINDINGS)                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ✅ BEST MODEL: Ensemble (XGBoost + SVM + LR)            ║
║     • Accuracy: 77.4% (20% test size, 1,065 samples)             ║
║     • Features: 10,000 TF-IDF (1-5 grams)                             ║
║                                                                              ║
║  📊 TEST SIZE IMPACT:                                               ║
║     • 3% test: 81.9% (160 samples) - High variance              ║
║     • 20% test: 77.4% (1,065 samples) - Reliable                 ║
║     → Larger test size = More honest evaluation               ║
║                                                                              ║
║  🔧 CÁC TECHNIQUE HIỆU QUẢ:                                              ║
║     • Vietnamese stopwords removal (+3-5%)                              ║
║     • TF-IDF n-grams (1-5) (+2-3%)                                  ║
║     • Class weight balancing (+5-8%)                                ║
║     • Ensemble of multiple models (+1-2%)                              ║
║                                                                              ║
║  ⚠️ CÁC HẠN CHẾ:                                                     ║
║     • Auto-labeling có noise (~10-15%)                                  ║
║     • Binary classification (lost NEUTRAL distinction)           ║
║     • Domain specific (financial news only)                            ║
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
