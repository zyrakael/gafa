import matplotlib.pyplot as plt
import numpy as np

# 使用markdown中的6个时间步数据
steps = np.arange(1, 7)

# 数据
history = np.array([2.1, 2.4, 2.0, 2.7, 2.5, 2.8])
x_raw = np.array([3.0, 3.4, 3.2, 3.8, 3.6, 4.1])
x_true = np.array([3.1, 3.0, 3.5, 3.7, 3.2, 4.0])
residuals = np.array([0.1, -0.4, 0.3, -0.1, -0.4, -0.1])
adapter_c = np.array([0.15, -0.50, 0.25, -0.30, -0.60, 0.10])
reliability_r = np.array([0.8, 0.6, 0.9, 0.2, 0.5, 0.1])
weighted_correction = np.array([0.12, -0.30, 0.225, -0.06, -0.30, 0.01])
x_align = np.array([3.12, 3.10, 3.425, 3.74, 3.30, 4.11])

# 颜色定义
color_history = '#1f77b4'      # 深蓝色
color_raw = '#ff7f0e'          # 橙色
color_true = '#2ca02c'         # 绿色
color_residual = '#d62728'     # 红色
color_align = '#9467bd'        # 紫色
color_adapter = '#FF6B6B'      # 浅红色
color_reliability = '#4ECDC4'  # 青色
color_gated = '#95E1D3'        # 浅绿色

print("生成5个折线图...")

# 1. 历史值 - 深蓝色
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(steps, history, 'o-', linewidth=3.5, markersize=8, color=color_history)
ax.set_ylim([history.min() - 0.3, history.max() + 0.3])
ax.axis('off')
plt.tight_layout()
plt.savefig('line_1_history.svg', format='svg', bbox_inches='tight', facecolor='white', pad_inches=0.1)
plt.close()
print("✓ line_1_history.svg (深蓝色)")

# 2. 原始预测值 - 橙色
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(steps, x_raw, 's-', linewidth=3.5, markersize=8, color=color_raw)
ax.set_ylim([x_raw.min() - 0.3, x_raw.max() + 0.3])
ax.axis('off')
plt.tight_layout()
plt.savefig('line_2_raw_forecast.svg', format='svg', bbox_inches='tight', facecolor='white', pad_inches=0.1)
plt.close()
print("✓ line_2_raw_forecast.svg (橙色)")

# 3. 真实值 - 绿色
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(steps, x_true, '^-', linewidth=3.5, markersize=8, color=color_true)
ax.set_ylim([x_true.min() - 0.3, x_true.max() + 0.3])
ax.axis('off')
plt.tight_layout()
plt.savefig('line_3_true_value.svg', format='svg', bbox_inches='tight', facecolor='white', pad_inches=0.1)
plt.close()
print("✓ line_3_true_value.svg (绿色)")

# 4. 残差值 - 红色
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(steps, residuals, 'v-', linewidth=3.5, markersize=8, color=color_residual)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.4)
ax.set_ylim([residuals.min() - 0.1, residuals.max() + 0.1])
ax.axis('off')
plt.tight_layout()
plt.savefig('line_4_residuals.svg', format='svg', bbox_inches='tight', facecolor='white', pad_inches=0.1)
plt.close()
print("✓ line_4_residuals.svg (红色)")

# 5. 最终对齐预测 - 紫色（体现补上去的部分）
fig, ax = plt.subplots(figsize=(12, 6))
# 绘制原始预测作为参考（虚线）
ax.plot(steps, x_raw, '--', linewidth=2, markersize=6, color=color_raw, alpha=0.4)
# 绘制最终对齐预测
ax.plot(steps, x_align, 'D-', linewidth=3.5, markersize=8, color=color_align)

# 用竖线体现补上去的部分
for i, (raw, align, corr) in enumerate(zip(x_raw, x_align, weighted_correction)):
    if corr > 0:
        # 向上补
        ax.plot([steps[i], steps[i]], [raw, align], linewidth=2.5, color='#2ca02c', alpha=0.7)
        ax.scatter([steps[i]], [align], s=50, color='#2ca02c', alpha=0.7, zorder=5)
    elif corr < 0:
        # 向下补
        ax.plot([steps[i], steps[i]], [raw, align], linewidth=2.5, color='#d62728', alpha=0.7)
        ax.scatter([steps[i]], [align], s=50, color='#d62728', alpha=0.7, zorder=5)

ax.set_ylim([min(x_raw.min(), x_align.min()) - 0.3, max(x_raw.max(), x_align.max()) + 0.3])
ax.axis('off')
plt.tight_layout()
plt.savefig('line_5_aligned_forecast.svg', format='svg', bbox_inches='tight', facecolor='white', pad_inches=0.1)
plt.close()
print("✓ line_5_aligned_forecast.svg (紫色，体现补上去的部分)")

print("\n生成3个柱状图...")

# 1. Adapter输出 - 浅红色
fig, ax = plt.subplots(figsize=(12, 6))
colors_adapter = [color_adapter if c >= 0 else '#FF9999' for c in adapter_c]
ax.bar(steps, adapter_c, color=colors_adapter, alpha=0.85, edgecolor='black', linewidth=1.2, width=0.65)
ax.axhline(y=0, color='black', linewidth=1.5, alpha=0.5)
ax.set_ylim([adapter_c.min() - 0.1, adapter_c.max() + 0.1])
ax.axis('off')
plt.tight_layout()
plt.savefig('bar_1_adapter_output.svg', format='svg', bbox_inches='tight', facecolor='white', pad_inches=0.1)
plt.close()
print("✓ bar_1_adapter_output.svg (浅红色 - Adapter输出 c)")

# 2. 门控权重 - 青色
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(steps, reliability_r, color=color_reliability, alpha=0.85, edgecolor='black', linewidth=1.2, width=0.65)
ax.set_ylim([0, 1.0])
ax.axis('off')
plt.tight_layout()
plt.savefig('bar_2_reliability_weight.svg', format='svg', bbox_inches='tight', facecolor='white', pad_inches=0.1)
plt.close()
print("✓ bar_2_reliability_weight.svg (青色 - 门控权重 r)")

# 3. 门控和Adapter的合并 - 浅绿色
fig, ax = plt.subplots(figsize=(12, 6))
colors_gated = [color_gated if wc >= 0 else '#FFB6B9' for wc in weighted_correction]
ax.bar(steps, weighted_correction, color=colors_gated, alpha=0.85, edgecolor='black', linewidth=1.2, width=0.65)
ax.axhline(y=0, color='black', linewidth=1.5, alpha=0.5)
ax.set_ylim([weighted_correction.min() - 0.05, weighted_correction.max() + 0.05])
ax.axis('off')
plt.tight_layout()
plt.savefig('bar_3_gated_correction.svg', format='svg', bbox_inches='tight', facecolor='white', pad_inches=0.1)
plt.close()
print("✓ bar_3_gated_correction.svg (浅绿色 - 门控和Adapter的合并 r⊙c)")

print("\n" + "="*70)
print("所有矢量图生成完成！")
print("="*70)
print("\n折线图（5个）：")
print("  1. line_1_history.svg - 历史值（深蓝色）")
print("  2. line_2_raw_forecast.svg - 原始预测值（橙色）")
print("  3. line_3_true_value.svg - 真实值（绿色）")
print("  4. line_4_residuals.svg - 残差值（红色）")
print("  5. line_5_aligned_forecast.svg - 最终对齐预测（紫色，体现补上去的部分）")
print("\n柱状图（3个）：")
print("  1. bar_1_adapter_output.svg - Adapter输出（浅红色）")
print("  2. bar_2_reliability_weight.svg - 门控权重（青色）")
print("  3. bar_3_gated_correction.svg - 门控和Adapter的合并（浅绿色）")
print("="*70)

# 生成数据统计信息
print("\n数据统计：")
print(f"历史值范围: [{history.min():.2f}, {history.max():.2f}]")
print(f"原始预测范围: [{x_raw.min():.2f}, {x_raw.max():.2f}]")
print(f"真实值范围: [{x_true.min():.2f}, {x_true.max():.2f}]")
print(f"残差范围: [{residuals.min():.2f}, {residuals.max():.2f}]")
print(f"Adapter输出范围: [{adapter_c.min():.2f}, {adapter_c.max():.2f}]")
print(f"门控权重范围: [{reliability_r.min():.2f}, {reliability_r.max():.2f}]")
print(f"门控修正范围: [{weighted_correction.min():.3f}, {weighted_correction.max():.3f}]")
print(f"最终预测范围: [{x_align.min():.2f}, {x_align.max():.2f}]")
