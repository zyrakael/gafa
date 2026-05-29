import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import io
from pathlib import Path

# 数据来自 markdown
history = [2.1, 2.4, 2.0, 2.7, 2.5, 2.8]
x_raw = [3.0, 3.4, 3.2, 3.8, 3.6, 4.1]
x_true = [3.1, 3.0, 3.5, 3.7, 3.2, 4.0]
residual_e = [0.1, -0.4, 0.3, -0.1, -0.4, -0.1]
adapter_c = [0.15, -0.50, 0.25, -0.30, -0.60, 0.10]
reliability_r = [0.8, 0.6, 0.9, 0.2, 0.5, 0.1]
weighted_correction = [0.12, -0.30, 0.225, -0.06, -0.30, 0.01]
x_align = [3.12, 3.10, 3.425, 3.74, 3.30, 4.11]

# 时间步
steps = np.arange(1, 7)
history_steps = np.arange(-5, 1)

# ============ 第一部分：折线图 ============
fig1, ax1 = plt.subplots(figsize=(14, 8))

# 绘制历史值
ax1.plot(history_steps, history, 'o-', linewidth=2.5, markersize=6, 
         color='#1f77b4', label='Historical Data', alpha=0.8)

# 绘制原始预测
ax1.plot(steps, x_raw, 's-', linewidth=2.5, markersize=6, 
         color='#ff7f0e', label='Raw Forecast', alpha=0.8)

# 绘制真实值
ax1.plot(steps, x_true, '^-', linewidth=2.5, markersize=6, 
         color='#2ca02c', label='Ground Truth', alpha=0.8)

# 绘制残差
ax1.plot(steps, residual_e, 'v-', linewidth=2.5, markersize=6, 
         color='#d62728', label='Residual (e)', alpha=0.8)

# 绘制最终对齐预测
ax1.plot(steps, x_align, 'D-', linewidth=2.5, markersize=6, 
         color='#9467bd', label='Aligned Forecast', alpha=0.8)

# 添加虚线连接原始预测和对齐预测，显示修正部分
for i, step in enumerate(steps):
    if weighted_correction[i] != 0:
        ax1.plot([step, step], [x_raw[i], x_align[i]], 'k--', alpha=0.3, linewidth=1)

ax1.set_xlabel('Time Step', fontsize=12, fontweight='bold')
ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
ax1.set_title('GAFA Algorithm: Time Series Forecasting with Residual Correction', 
              fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax1.text(0.5, ax1.get_ylim()[1]*0.95, 'Future →', ha='center', fontsize=10, alpha=0.7)

plt.tight_layout()
plt.savefig('gafa_timeseries_lines.svg', format='svg', dpi=300, bbox_inches='tight')
plt.savefig('gafa_timeseries_lines.png', format='png', dpi=300, bbox_inches='tight')
print("✓ Saved: gafa_timeseries_lines.svg & .png")
plt.close()

# ============ 第二部分：柱状图 ============
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))

# 颜色定义
color_adapter = '#FF6B6B'      # 红色
color_reliability = '#4ECDC4'  # 青色
color_weighted = '#95E1D3'     # 浅绿色

# 1. Adapter 输出 (c)
ax = axes[0]
colors_c = [color_adapter if c >= 0 else '#FF6B6B' for c in adapter_c]
bars1 = ax.bar(steps, adapter_c, color=colors_c, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Time Step', fontsize=11, fontweight='bold')
ax.set_ylabel('Correction Value', fontsize=11, fontweight='bold')
ax.set_title('Adapter Output: c', fontsize=12, fontweight='bold', pad=15)
ax.set_ylim([-0.7, 0.3])
ax.grid(True, alpha=0.2, axis='y', linestyle='--')
ax.set_xticks(steps)

# 添加数值标签
for i, (step, val) in enumerate(zip(steps, adapter_c)):
    ax.text(step, val + (0.03 if val >= 0 else -0.05), f'{val:.2f}', 
            ha='center', va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')

# 2. Reliability 权重 (r)
ax = axes[1]
bars2 = ax.bar(steps, reliability_r, color=color_reliability, alpha=0.8, 
               edgecolor='black', linewidth=1.5)
ax.set_xlabel('Time Step', fontsize=11, fontweight='bold')
ax.set_ylabel('Weight Value', fontsize=11, fontweight='bold')
ax.set_title('Reliability Weight: r', fontsize=12, fontweight='bold', pad=15)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.2, axis='y', linestyle='--')
ax.set_xticks(steps)

# 添加数值标签
for step, val in zip(steps, reliability_r):
    ax.text(step, val + 0.03, f'{val:.1f}', ha='center', va='bottom', 
            fontsize=9, fontweight='bold')

# 3. 加权修正 (r ⊙ c)
ax = axes[2]
colors_weighted = [color_weighted if wc >= 0 else '#FFB6B9' for wc in weighted_correction]
bars3 = ax.bar(steps, weighted_correction, color=colors_weighted, alpha=0.8, 
               edgecolor='black', linewidth=1.5)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Time Step', fontsize=11, fontweight='bold')
ax.set_ylabel('Weighted Correction', fontsize=11, fontweight='bold')
ax.set_title('Gated Correction: r ⊙ c', fontsize=12, fontweight='bold', pad=15)
ax.set_ylim([-0.4, 0.3])
ax.grid(True, alpha=0.2, axis='y', linestyle='--')
ax.set_xticks(steps)

# 添加数值标签
for i, (step, val) in enumerate(zip(steps, weighted_correction)):
    ax.text(step, val + (0.02 if val >= 0 else -0.03), f'{val:.3f}', 
            ha='center', va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')

plt.suptitle('GAFA Algorithm: Adapter, Reliability Weight, and Gated Correction', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('gafa_bars_components.svg', format='svg', dpi=300, bbox_inches='tight')
plt.savefig('gafa_bars_components.png', format='png', dpi=300, bbox_inches='tight')
print("✓ Saved: gafa_bars_components.svg & .png")
plt.close()

# ============ 第三部分：综合对比图 ============
fig3, ax = plt.subplots(figsize=(14, 7))

# 绘制原始预测和真实值
ax.plot(steps, x_raw, 'o-', linewidth=2.5, markersize=7, 
        color='#FF6B6B', label='Raw Forecast', alpha=0.8)
ax.plot(steps, x_true, 's-', linewidth=2.5, markersize=7, 
        color='#2ca02c', label='Ground Truth', alpha=0.8)
ax.plot(steps, x_align, '^-', linewidth=2.5, markersize=7, 
        color='#9467bd', label='Aligned Forecast', alpha=0.8)

# 填充修正区域
for i in range(len(steps)):
    if weighted_correction[i] > 0:
        ax.fill_between([steps[i]-0.2, steps[i]+0.2], x_raw[i], x_align[i], 
                        alpha=0.2, color='green', label='Upward Correction' if i == 0 else '')
    elif weighted_correction[i] < 0:
        ax.fill_between([steps[i]-0.2, steps[i]+0.2], x_raw[i], x_align[i], 
                        alpha=0.2, color='red', label='Downward Correction' if i == 1 else '')

ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Value', fontsize=12, fontweight='bold')
ax.set_title('GAFA: Raw Forecast vs Aligned Forecast vs Ground Truth', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(steps)

plt.tight_layout()
plt.savefig('gafa_comparison.svg', format='svg', dpi=300, bbox_inches='tight')
plt.savefig('gafa_comparison.png', format='png', dpi=300, bbox_inches='tight')
print("✓ Saved: gafa_comparison.svg & .png")
plt.close()

print("\n" + "="*60)
print("All visualizations generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. gafa_timeseries_lines.svg/png - Time series with all components")
print("  2. gafa_bars_components.svg/png - Adapter, Reliability, Gated Correction")
print("  3. gafa_comparison.svg/png - Comparison of forecasts")
