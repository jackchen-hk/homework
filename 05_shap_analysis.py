# -*- coding: utf-8 -*-
"""
=================================================================
  基于多维度特征的劳动力AI替代风险挖掘与高危岗位识别
  Step 5: SHAP可解释性分析 (成员B负责)
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局配置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

RANDOM_STATE = 42
CLASS_NAMES = ['Low Risk', 'Medium Risk', 'High Risk']

print("=" * 70)
print("  第五步：SHAP可解释性分析")
print("=" * 70)

# ============================================================
# 1. 加载数据和模型
# ============================================================
print("\n【1】加载数据和模型...")

X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train_filled.csv'))
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test_filled.csv'))
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test_cls_eq.csv')).squeeze()

with open(os.path.join(DATA_DIR, 'feature_columns.pkl'), 'rb') as f:
    feature_cols = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'lightgbm_tuned.pkl'), 'rb') as f:
    model = pickle.load(f)

print(f"  训练集: {X_train.shape}")
print(f"  测试集: {X_test.shape}")

# ============================================================
# 2. 计算SHAP值
# ============================================================
print("\n【2】计算SHAP值 (TreeExplainer)...")

explainer = shap.TreeExplainer(model)

# 使用测试集的一个子集计算SHAP值（全量太慢）
shap_sample_size = min(2000, len(X_test))
X_sample = X_test.iloc[:shap_sample_size]

shap_values = explainer.shap_values(X_sample)
print(f"  SHAP值计算完成, 样本数: {X_sample.shape[0]}")

# 检查SHAP值形状并处理
if isinstance(shap_values, list):
    # 多分类: list of arrays
    shap_array = np.array(shap_values)  # (n_classes, n_samples, n_features)
    print(f"  SHAP值形状 (多分类): {shap_array.shape}")
elif len(shap_values.shape) == 3:
    # 已经是 (n_samples, n_features, n_classes)
    shap_array = shap_values.transpose(2, 0, 1)  # -> (n_classes, n_samples, n_features)
    print(f"  SHAP值形状 (转置后): {shap_array.shape}")
else:
    shap_array = shap_values
    print(f"  SHAP值形状: {shap_array.shape}")

# 保存SHAP值
np.savez(os.path.join(OUTPUT_DIR, 'shap_values.npz'),
         shap_values=shap_array,
         X_sample=X_sample.values,
         feature_names=feature_cols)
print(f"  ✓ SHAP值已保存: shap_values.npz")

# ============================================================
# 3. 全局特征重要性 (Summary Plot)
# ============================================================
print("\n【3】全局特征重要性分析...")

# 图23: SHAP Summary Plot - High Risk类
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_array[2], X_sample, feature_names=feature_cols,
                  max_display=20, show=False, plot_size=(12, 8))
plt.title('图23: SHAP特征重要性 - High Risk类别', fontsize=14, pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig23_shap_summary_high_risk.png'))
plt.close()
print(">>> 图23已保存: fig23_shap_summary_high_risk.png")

# 图24: SHAP Summary Plot - 所有类别对比
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
for i, name in enumerate(CLASS_NAMES):
    plt.sca(axes[i])
    shap.summary_plot(shap_array[i], X_sample, feature_names=feature_cols,
                      max_display=15, show=False, plot_size=None)
    axes[i].set_title(f'{name}', fontsize=12)
plt.suptitle('图24: SHAP特征重要性 - 三类对比', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig24_shap_summary_all_classes.png'))
plt.close()
print(">>> 图24已保存: fig24_shap_summary_all_classes.png")

# 图25: SHAP Bar Plot - 平均绝对SHAP值
fig, ax = plt.subplots(figsize=(12, 8))
# 计算所有类别的平均绝对SHAP
mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_array], axis=0)
shap_importance = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=True)

top_features = shap_importance.tail(20)
ax.barh(top_features['feature'], top_features['mean_abs_shap'],
        color='steelblue', alpha=0.8, edgecolor='white')
ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
ax.set_title('图25: 全局特征重要性 (平均|SHAP值|)', fontsize=14)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig25_shap_bar_importance.png'))
plt.close()
print(">>> 图25已保存: fig25_shap_bar_importance.png")

# 保存SHAP重要性排序
shap_importance.sort_values('mean_abs_shap', ascending=False).to_csv(
    os.path.join(OUTPUT_DIR, 'shap_feature_importance.csv'), index=False)

# ============================================================
# 4. 局部特征归因 (Waterfall Plot)
# ============================================================
print("\n【4】局部特征归因分析...")

# 找一个高危岗位的例子
y_pred = model.predict(X_test)
high_risk_mask = y_pred == 2
if high_risk_mask.any():
    high_risk_idx = np.where(high_risk_mask)[0][0]
else:
    high_risk_idx = 0

# 图26: Waterfall Plot - 高危岗位
fig, ax = plt.subplots(figsize=(12, 8))
shap.waterfall_plot(shap.Explanation(
    values=shap_values[2][high_risk_idx],
    base_values=explainer.expected_value[2],
    data=X_sample.iloc[high_risk_idx].values,
    feature_names=feature_cols
), max_display=15, show=False)
plt.title('图26: 高危岗位SHAP归因 (Waterfall Plot)', fontsize=14, pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig26_shap_waterfall_high_risk.png'))
plt.close()
print(">>> 图26已保存: fig26_shap_waterfall_high_risk.png")

# 找一个低危岗位的例子
low_risk_mask = y_pred == 0
if low_risk_mask.any():
    low_risk_idx = np.where(low_risk_mask)[0][0]
else:
    low_risk_idx = 1

# 图27: Waterfall Plot - 低危岗位
fig, ax = plt.subplots(figsize=(12, 8))
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0][low_risk_idx],
    base_values=explainer.expected_value[0],
    data=X_sample.iloc[low_risk_idx].values,
    feature_names=feature_cols
), max_display=15, show=False)
plt.title('图27: 低危岗位SHAP归因 (Waterfall Plot)', fontsize=14, pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig27_shap_waterfall_low_risk.png'))
plt.close()
print(">>> 图27已保存: fig27_shap_waterfall_low_risk.png")

# ============================================================
# 5. SHAP Dependence Plot (关键特征依赖)
# ============================================================
print("\n【5】SHAP依赖图...")

# 选择Top 4特征
top4_features = shap_importance.tail(4)['feature'].values
print(f"  Top 4特征: {list(top4_features)}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, feat in enumerate(top4_features):
    ax = axes[idx // 2][idx % 2]
    feat_idx = list(feature_cols).index(feat)
    shap.dependence_plot(feat_idx, shap_array[2], X_sample,
                         feature_names=feature_cols, ax=ax, show=False)
    ax.set_title(f'SHAP依赖图: {feat}', fontsize=12)

plt.suptitle('图28: 关键特征SHAP依赖图 (High Risk)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig28_shap_dependence.png'))
plt.close()
print(">>> 图28已保存: fig28_shap_dependence.png")

# ============================================================
# 6. SHAP Force Plot (交互式)
# ============================================================
print("\n【6】SHAP Force Plot...")

# 选择5个高危岗位和5个低危岗位
high_risk_indices = np.where(y_pred[:shap_sample_size] == 2)[0][:5]
low_risk_indices = np.where(y_pred[:shap_sample_size] == 0)[0][:5]
selected_indices = np.concatenate([high_risk_indices, low_risk_indices])

# 保存Force Plot HTML
try:
    shap_html = shap.force_plot(
        explainer.expected_value[2],
        shap_values[2][selected_indices],
        X_sample.iloc[selected_indices],
        feature_names=feature_cols,
        show=False
    )
    with open(os.path.join(OUTPUT_DIR, 'shap_force_plot.html'), 'w') as f:
        f.write(shap.getjs())
        f.write(shap_html.html())
    print(f"  ✓ Force Plot已保存: shap_force_plot.html")
except Exception as e:
    print(f"  ⚠ Force Plot保存失败: {e}")

# ============================================================
# 7. 高危 vs 低危 SHAP对比
# ============================================================
print("\n【7】高危 vs 低危 SHAP特征贡献对比...")

high_risk_mask = y_pred[:shap_sample_size] == 2
low_risk_mask = y_pred[:shap_sample_size] == 0

high_risk_shap = np.abs(shap_array[2][high_risk_mask]).mean(axis=0)
low_risk_shap = np.abs(shap_array[0][low_risk_mask]).mean(axis=0)

shap_compare = pd.DataFrame({
    'feature': feature_cols,
    'high_risk_abs_shap': high_risk_shap,
    'low_risk_abs_shap': low_risk_shap,
    'shap_diff': high_risk_shap - low_risk_shap
}).sort_values('shap_diff', ascending=False)

print(f"\n--- 高危岗位更依赖的特征 (SHAP差异) ---")
for _, row in shap_compare.head(10).iterrows():
    print(f"  {row['feature']:40s}  高危|SHAP|={row['high_risk_abs_shap']:.4f}  低危|SHAP|={row['low_risk_abs_shap']:.4f}  差异={row['shap_diff']:+.4f}")

shap_compare.to_csv(os.path.join(OUTPUT_DIR, 'shap_high_vs_low_comparison.csv'), index=False)

# 图29: 高危vs低危SHAP对比
fig, ax = plt.subplots(figsize=(14, 8))
top_diff = shap_compare.reindex(shap_compare['shap_diff'].abs().sort_values(ascending=False).index).head(15)
y_pos = range(len(top_diff))

ax.barh(y_pos, top_diff['high_risk_abs_shap'], height=0.4, label='High Risk', color='#e74c3c', alpha=0.8, align='center')
ax.barh([y + 0.4 for y in y_pos], top_diff['low_risk_abs_shap'], height=0.4, label='Low Risk', color='#2ecc71', alpha=0.8, align='center')
ax.set_yticks([y + 0.2 for y in y_pos])
ax.set_yticklabels(top_diff['feature'], fontsize=10)
ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
ax.set_title('图29: 高危 vs 低危岗位 SHAP特征贡献对比', fontsize=14)
ax.legend(fontsize=11)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig29_shap_high_vs_low.png'))
plt.close()
print(">>> 图29已保存: fig29_shap_high_vs_low.png")

# ============================================================
# 8. 总结
# ============================================================
print("\n" + "=" * 70)
print("  SHAP可解释性分析完成!")
print("=" * 70)

print(f"""
  【SHAP分析总结】

  1. Top 5 关键特征 (平均|SHAP|):
     {chr(10).join([f'     {i+1}. {row["feature"]:35s} |SHAP|={row["mean_abs_shap"]:.4f}' for i, (_, row) in enumerate(shap_importance.tail(5).iterrows())])}

  2. 高危岗位关键驱动因素:
     {chr(10).join([f'     - {row["feature"]} (差异={row["shap_diff"]:+.4f})' for _, row in shap_compare.head(5).iterrows()])}

  3. 生成文件:
     - outputs/shap_values.npz
     - outputs/shap_feature_importance.csv
     - outputs/shap_high_vs_low_comparison.csv
     - outputs/fig23-29 共7张图
     - outputs/shap_force_plot.html
""")
