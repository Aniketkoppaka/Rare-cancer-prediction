"""
Generates publication-quality diagnostic trajectory graphs and performance matrix plots.
Outputs:
1. figures/figure3.png (and figures/fig2_trajectory_and_policies.png)
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# Data from Continuous Q=0..25 Benchmark (Table 4 in Manuscript)
q_vals = list(range(26))
hits1 = [68.62, 74.73, 78.46, 80.85, 82.98, 84.52, 86.17, 87.29, 88.35, 89.20, 89.95, 
         90.48, 90.90, 91.22, 91.54, 91.81, 91.97, 92.13, 92.29, 92.39, 92.50, 92.61, 
         92.71, 92.77, 92.82, 92.87]
hits5 = [80.90, 82.71, 85.11, 86.76, 87.98, 88.94, 89.57, 90.05, 90.48, 90.85, 91.12,
         91.49, 91.86, 92.18, 92.50, 92.77, 92.93, 93.03, 93.14, 93.24, 93.30, 93.35,
         93.40, 93.46, 93.46, 93.51]
mrr = [0.7399, 0.7819, 0.8143, 0.8338, 0.8520, 0.8659, 0.8780, 0.8872, 0.8948, 0.9008, 0.9057,
       0.9102, 0.9140, 0.9172, 0.9201, 0.9226, 0.9242, 0.9256, 0.9268, 0.9279, 0.9288, 0.9296,
       0.9303, 0.9309, 0.9314, 0.9319]

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

# Annotations on ax1 matching Table 4 exactly
ax1.annotate(f'Q=5: {hits1[5]:.2f}%', xy=(5, hits1[5]), xytext=(5.5, 78),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5), fontsize=10, fontweight='bold')
ax1.annotate(f'Q=15: {hits1[15]:.2f}%', xy=(15, hits1[15]), xytext=(11.5, 85),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5), fontsize=10, fontweight='bold')
ax1.annotate(f'Q=25: {hits1[25]:.2f}%', xy=(25, hits1[25]), xytext=(19.2, 94.7),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5), fontsize=10, fontweight='bold')

# Plot 2: Performance Comparison Matrix (Bar / Heatmap style) matching Section 5.3 & Table 4
policies = ['Static (A1)', 'Highest-IC (Q=5)', 'Random (Q=5)', 'Margin-Boost (Q=5)', 'Policy A (Q=5)', 'Adaptive (Q=15)', 'Adaptive (Q=25)']
hits1_bar = [68.62, 77.18, 80.43, 84.52, 85.00, 91.81, 92.87]
colors = ['#7f7f7f', '#c7c7c7', '#aec7e8', '#4a90d9', '#1f77b4', '#17becf', '#005599']

bars = ax2.bar(policies, hits1_bar, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
ax2.set_title('Acquisition Policy & Scaling Comparison Matrix', fontsize=14, fontweight='bold', pad=12)
ax2.set_ylabel('Zero-Shot Hits@1 Accuracy (%)', fontsize=12)
ax2.set_ylim(60, 100)
ax2.grid(axis='y', linestyle='--', alpha=0.6)
plt.setp(ax2.get_xticklabels(), rotation=25, ha='right', fontsize=10)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.8,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figure3.png', dpi=300)
plt.savefig('figures/trajectory_and_performance_matrix.png', dpi=300)
print("Saved figures/figure3.png and figures/trajectory_and_performance_matrix.png successfully!")
