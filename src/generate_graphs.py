"""
Generates publication-quality diagnostic trajectory graphs and performance matrix plots.
Outputs:
1. trajectory_curve_q0_to_25.png
2. confusion_performance_matrix.png
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# Data from Continuous Q=0..25 Benchmark
q_vals = list(range(26))
hits1 = [68.62, 74.68, 77.77, 80.74, 83.35, 84.73, 87.02, 87.77, 88.94, 89.47, 90.00, 
         90.69, 91.06, 91.22, 91.33, 91.86, 92.07, 92.18, 92.61, 92.77, 92.93, 93.03, 
         93.30, 93.46, 93.56, 93.62]
hits5 = [80.90, 82.71, 84.10, 86.12, 87.71, 88.40, 89.47, 89.95, 90.53, 90.85, 91.12,
         91.49, 91.76, 91.97, 92.39, 92.71, 92.82, 92.98, 93.09, 93.19, 93.46, 93.67,
         93.72, 93.83, 93.83, 94.04]
mrr = [0.7399, 0.7818, 0.8081, 0.8316, 0.8524, 0.8647, 0.8815, 0.8885, 0.8969, 0.9020, 0.9069,
       0.9117, 0.9146, 0.9168, 0.9187, 0.9225, 0.9244, 0.9261, 0.9290, 0.9304, 0.9321, 0.9340,
       0.9357, 0.9370, 0.9379, 0.9386]

# Plot 1: Continuous Trajectory Curve
ax1.plot(q_vals, hits1, marker='o', color='#1f77b4', linewidth=2.5, label='Top-1 Accuracy (Hits@1)')
ax1.plot(q_vals, hits5, marker='s', color='#2ca02c', linewidth=2.0, linestyle='--', label='Top-5 Accuracy (Hits@5)')
ax1.axhline(68.62, color='#d62728', linestyle=':', label='Static Baseline (Q=0, 68.62%)')
ax1.axvline(5, color='#9467bd', linestyle='-.', alpha=0.7, label='Standard Inquiry Cap (Q=5)')
ax1.axvline(15, color='#ff7f0e', linestyle='-.', alpha=0.7, label='Deep Search Cap (Q=15)')

ax1.set_title('Continuous Diagnostic Accuracy Scaling (Q = 0 to 25)', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Inquiry Budget / Number of Clinician Questions (Q)', fontsize=12)
ax1.set_ylabel('Accuracy on Locked Test Cohort (%)', fontsize=12)
ax1.set_ylim(65, 96)
ax1.set_xlim(0, 25)
ax1.legend(loc='lower right', frameon=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)

# Annotations on ax1
ax1.annotate(f'Q=5: {hits1[5]:.1f}%', xy=(5, hits1[5]), xytext=(5.5, 78),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5), fontsize=10, fontweight='bold')
ax1.annotate(f'Q=15: {hits1[15]:.1f}%', xy=(15, hits1[15]), xytext=(12, 85),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5), fontsize=10, fontweight='bold')
ax1.annotate(f'Q=25: {hits1[25]:.1f}%', xy=(25, hits1[25]), xytext=(20, 95),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5), fontsize=10, fontweight='bold')

# Plot 2: Performance Comparison Matrix (Bar / Heatmap style)
policies = ['Static (A1)', 'Random (Q=5)', 'Highest-IC (Q=5)', 'Adaptive (Q=5)', 'Adaptive (Q=10)', 'Adaptive (Q=25)']
hits1_bar = [68.62, 80.11, 77.50, 84.73, 90.00, 93.62]
colors = ['#7f7f7f', '#aec7e8', '#c7c7c7', '#1f77b4', '#17becf', '#005599']

bars = ax2.bar(policies, hits1_bar, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
ax2.set_title('Acquisition Policy & Scaling Comparison Matrix', fontsize=14, fontweight='bold', pad=12)
ax2.set_ylabel('Zero-Shot Hits@1 Accuracy (%)', fontsize=12)
ax2.set_ylim(60, 100)
ax2.grid(axis='y', linestyle='--', alpha=0.6)
plt.setp(ax2.get_xticklabels(), rotation=25, ha='right', fontsize=10)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.8,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/trajectory_and_performance_matrix.png', dpi=300)
print("Saved figures/trajectory_and_performance_matrix.png successfully!")
