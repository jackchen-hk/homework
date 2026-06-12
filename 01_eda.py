# -*- coding: utf-8 -*-
"""
=================================================================
  基于多维度特征的劳动力AI替代风险挖掘与高危岗位识别
  Step 1: 数据探索性分析 (EDA)
  负责人: 成员A
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'ai_job_trends_dataset.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 数据加载与基本信息
# ============================================================
print("=" * 70)
print("  第一部分：数据基本信息")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\n数据形状: {df.shape[0]} 行 × {df.shape[1]} 列\n")

print("--- 列名与数据类型 ---")
print(df.dtypes)
print()

print("--- 前5行预览 ---")
print(df.head())
print()

print("--- 数据基本信息 ---")
print(df.info())
print()

print("--- 缺失值统计 ---")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'缺失数': missing, '缺失比例(%)': missing_pct})
print(missing_df)
print()

print("--- 数值型特征描述统计 ---")
print(df.describe().round(2))
print()

print("--- 分类型特征描述统计 ---")
print(df.describe(include='object').round(2))
print()

print("--- 重复行检测 ---")
print(f"完全重复行: {df.duplicated().sum()} ({df.duplicated().sum()/len(df)*100:.2f}%)")
print()

# ============================================================
# 2. 目标变量分析
# ============================================================
print("=" * 70)
print("  第二部分：目标变量 Automation Risk (%) 分析")
print("=" * 70)

target = 'Automation Risk (%)'

print(f"\n--- {target} 描述统计 ---")
print(df[target].describe().round(4))
print(f"偏度(Skewness): {df[target].skew():.4f}")
print(f"峰度(Kurtosis): {df[target].kurtosis():.4f}")
print()

# 风险分级
df['Risk_Category'] = pd.cut(df[target], bins=[-0.01, 30, 60, 100],
                             labels=['Low Risk (0-30%)', 'Medium Risk (30-60%)', 'High Risk (60-100%)'])
risk_dist = df['Risk_Category'].value_counts().sort_index()
risk_pct = (risk_dist / len(df) * 100).round(2)
print("--- 风险等级分布 ---")
for cat, count in risk_dist.items():
    print(f"  {cat}: {count} 条 ({risk_pct[cat]:.2f}%)")
print()

# 图1: 目标变量分布
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1a. 直方图 + KDE
axes[0].hist(df[target], bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
df[target].plot.kde(ax=axes[0], color='red', linewidth=2)
axes[0].axvline(df[target].mean(), color='orange', linestyle='--', linewidth=1.5, label=f'Mean={df[target].mean():.1f}')
axes[0].axvline(df[target].median(), color='green', linestyle='--', linewidth=1.5, label=f'Median={df[target].median():.1f}')
axes[0].set_xlabel('Automation Risk (%)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('自动化替代风险概率密度分布', fontsize=14)
axes[0].legend(fontsize=10)

# 1b. 风险等级饼图
colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']
explode = (0.02, 0.02, 0.05)
axes[1].pie(risk_dist, labels=risk_dist.index, autopct='%1.1f%%',
            colors=colors_pie, explode=explode, startangle=90, textprops={'fontsize': 10})
axes[1].set_title('风险等级分布', fontsize=14)

# 1c. 累积分布函数 (CDF)
sorted_risk = np.sort(df[target])
cdf = np.arange(1, len(sorted_risk) + 1) / len(sorted_risk)
axes[2].plot(sorted_risk, cdf, linewidth=2, color='steelblue')
axes[2].axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
axes[2].axvline(x=df[target].median(), color='green', linestyle='--', alpha=0.7, label=f'Median={df[target].median():.1f}')
axes[2].set_xlabel('Automation Risk (%)', fontsize=12)
axes[2].set_ylabel('Cumulative Probability', fontsize=12)
axes[2].set_title('自动化风险累积分布函数(CDF)', fontsize=14)
axes[2].legend(fontsize=10)

plt.suptitle('图1: 目标变量 Automation Risk 整体分布', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig01_target_distribution.png'))
plt.close()
print(">>> 图1已保存: fig01_target_distribution.png")

# ============================================================
# 3. 数值特征分布分析
# ============================================================
print("\n" + "=" * 70)
print("  第三部分：数值特征分布分析")
print("=" * 70)

numeric_cols = ['Median Salary (USD)', 'Experience Required (Years)',
                'Job Openings (2024)', 'Projected Openings (2030)',
                'Remote Work Ratio (%)', 'Gender Diversity (%)']

# 图2: 数值特征箱线图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, col in enumerate(numeric_cols):
    ax = axes[idx // 3][idx % 3]
    bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='steelblue'),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='steelblue'),
                    capprops=dict(color='steelblue'))
    ax.set_title(col, fontsize=11)
    ax.set_ylabel('Value', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('图2: 数值特征箱线图(异常值检测)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig02_numeric_boxplots.png'))
plt.close()
print(">>> 图2已保存: fig02_numeric_boxplots.png")

# 图3: 数值特征直方图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, col in enumerate(numeric_cols):
    ax = axes[idx // 3][idx % 3]
    ax.hist(df[col], bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean={df[col].mean():.1f}')
    ax.axvline(df[col].median(), color='green', linestyle='--', label=f'Median={df[col].median():.1f}')
    ax.set_title(col, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('图3: 数值特征频率分布', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig03_numeric_histograms.png'))
plt.close()
print(">>> 图3已保存: fig03_numeric_histograms.png")

# ============================================================
# 4. 分类特征分布分析
# ============================================================
print("\n" + "=" * 70)
print("  第四部分：分类特征分布分析")
print("=" * 70)

categorical_cols = ['Industry', 'AI Impact Level', 'Required Education',
                    'Job Status', 'Location']

# 图4: 分类特征计数图
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
for idx, col in enumerate(categorical_cols):
    ax = axes[idx // 3][idx % 3]
    order = df[col].value_counts().index
    sns.countplot(data=df, y=col, ax=ax, order=order, palette='Set2')
    ax.set_title(f'{col} 分布', fontsize=12)
    ax.set_xlabel('Count', fontsize=10)
    # 添加计数标注
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 50, p.get_y() + p.get_height() / 2,
                f'{int(width)}', ha='left', va='center', fontsize=9)

# 隐藏多余子图
axes[1][2].axis('off')
plt.suptitle('图4: 分类特征频数分布', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig04_categorical_counts.png'))
plt.close()
print(">>> 图4已保存: fig04_categorical_counts.png")

# ============================================================
# 5. 相关性分析
# ============================================================
print("\n" + "=" * 70)
print("  第五部分：相关性分析")
print("=" * 70)

# 图5: Pearson相关矩阵
numeric_all = numeric_cols + [target]
corr_matrix = df[numeric_all].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title('图5: 数值特征Pearson相关矩阵', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig05_correlation_matrix.png'))
plt.close()
print(">>> 图5已保存: fig05_correlation_matrix.png")

# 图6: Spearman秩相关矩阵
spearman_corr = df[numeric_all].corr(method='spearman')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(spearman_corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title('图6: 数值特征Spearman秩相关矩阵', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig06_spearman_correlation.png'))
plt.close()
print(">>> 图6已保存: fig06_spearman_correlation.png")

# 打印与目标变量的相关系数
print("\n--- 与Automation Risk (%)的相关系数 ---")
print("\n  [Pearson线性相关]")
pearson_with_target = corr_matrix[target].drop(target).sort_values(ascending=False)
for feat, val in pearson_with_target.items():
    sig = '***' if abs(val) > 0.3 else '**' if abs(val) > 0.2 else '*' if abs(val) > 0.1 else ''
    print(f"    {feat:40s}  r = {val:+.4f} {sig}")

print("\n  [Spearman秩相关(非线性)]")
spearman_with_target = spearman_corr[target].drop(target).sort_values(ascending=False)
for feat, val in spearman_with_target.items():
    sig = '***' if abs(val) > 0.3 else '**' if abs(val) > 0.2 else '*' if abs(val) > 0.1 else ''
    print(f"    {feat:40s}  ρ = {val:+.4f} {sig}")

# ============================================================
# 6. 分类特征与目标变量的关系
# ============================================================
print("\n" + "=" * 70)
print("  第六部分：分类特征 vs Automation Risk")
print("=" * 70)

# 图7: 各分类特征下Automation Risk的箱线图
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

for idx, col in enumerate(categorical_cols):
    ax = axes[idx // 3][idx % 3]
    order = df.groupby(col)[target].median().sort_values(ascending=False).index
    sns.boxplot(data=df, y=col, x=target, ax=ax, order=order, palette='Set3')
    ax.set_title(f'{col} vs Automation Risk', fontsize=12)
    ax.set_xlabel('Automation Risk (%)', fontsize=10)
    ax.set_ylabel(col, fontsize=10)

# 隐藏多余子图
axes[1][2].axis('off')
plt.suptitle('图7: 分类特征与自动化风险的箱线图对比', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig07_categorical_vs_risk_boxplot.png'))
plt.close()
print(">>> 图7已保存: fig07_categorical_vs_risk_boxplot.png")

# 图8: 各分类特征下风险等级的堆叠条形图
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
risk_colors = {'Low Risk (0-30%)': '#2ecc71', 'Medium Risk (30-60%)': '#f39c12', 'High Risk (60-100%)': '#e74c3c'}

for idx, col in enumerate(categorical_cols):
    ax = axes[idx // 3][idx % 3]
    ct = pd.crosstab(df[col], df['Risk_Category'], normalize='index') * 100
    ct = ct[['Low Risk (0-30%)', 'Medium Risk (30-60%)', 'High Risk (60-100%)']]
    ct.plot(kind='barh', stacked=True, ax=ax, color=[risk_colors[c] for c in ct.columns], edgecolor='white')
    ax.set_title(f'{col} 风险等级占比', fontsize=12)
    ax.set_xlabel('Percentage (%)', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.3)

axes[1][2].axis('off')
plt.suptitle('图8: 各分类维度下风险等级堆叠占比', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig08_risk_proportion_by_category.png'))
plt.close()
print(">>> 图8已保存: fig08_risk_proportion_by_category.png")

# ============================================================
# 7. 统计显著性检验
# ============================================================
print("\n" + "=" * 70)
print("  第七部分：统计显著性检验")
print("=" * 70)

print("\n--- Kruskal-Wallis 检验(分类特征 vs Automation Risk) ---")
print("  H0: 各组的中位数无显著差异\n")
for col in categorical_cols:
    groups = [g[target].values for _, g in df.groupby(col)]
    stat, p = stats.kruskal(*groups)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    print(f"  {col:25s}  H={stat:10.2f}  p={p:.2e}  {sig}")

print("\n--- Mann-Whitney U 检验(Job Status: Increasing vs Decreasing) ---")
inc = df[df['Job Status'] == 'Increasing'][target]
dec = df[df['Job Status'] == 'Decreasing'][target]
u_stat, p_val = stats.mannwhitneyu(inc, dec, alternative='two-sided')
print(f"  Increasing: median={inc.median():.2f}, mean={inc.mean():.2f}")
print(f"  Decreasing: median={dec.median():.2f}, mean={dec.mean():.2f}")
print(f"  U={u_stat:.2f}, p={p_val:.4f} {'显著' if p_val < 0.05 else '不显著'}")

# ============================================================
# 8. 多维交叉分析
# ============================================================
print("\n" + "=" * 70)
print("  第八部分：多维交叉分析")
print("=" * 70)

# 图9: Industry × AI Impact Level → 平均Automation Risk 热力图
pivot1 = df.pivot_table(values=target, index='Industry', columns='AI Impact Level', aggfunc='mean')
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, linewidths=0.5)
ax.set_title('图9: 行业 × AI冲击等级 → 平均自动化风险热力图', fontsize=14)
ax.set_ylabel('Industry', fontsize=12)
ax.set_xlabel('AI Impact Level', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig09_industry_impact_heatmap.png'))
plt.close()
print(">>> 图9已保存: fig09_industry_impact_heatmap.png")

# 图10: Education × Job Status → 平均Automation Risk 热力图
edu_order = ['High School', 'Associate Degree', "Bachelor's Degree", "Master's Degree", 'PhD']
pivot2 = df.pivot_table(values=target, index='Required Education', columns='Job Status', aggfunc='mean')
pivot2 = pivot2.reindex(edu_order)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, linewidths=0.5)
ax.set_title('图10: 学历 × 岗位趋势 → 平均自动化风险热力图', fontsize=14)
ax.set_ylabel('Required Education', fontsize=12)
ax.set_xlabel('Job Status', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_education_status_heatmap.png'))
plt.close()
print(">>> 图10已保存: fig10_education_status_heatmap.png")

# 图11: Location × Industry → 平均Automation Risk 热力图
pivot3 = df.pivot_table(values=target, index='Location', columns='Industry', aggfunc='mean')
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(pivot3, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, linewidths=0.5)
ax.set_title('图11: 地区 × 行业 → 平均自动化风险热力图', fontsize=14)
ax.set_ylabel('Location', fontsize=12)
ax.set_xlabel('Industry', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig11_location_industry_heatmap.png'))
plt.close()
print(">>> 图11已保存: fig11_location_industry_heatmap.png")

# ============================================================
# 9. 高危岗位识别初步
# ============================================================
print("\n" + "=" * 70)
print("  第九部分：高危岗位初步识别")
print("=" * 70)

high_risk = df[df[target] > 80].copy()
low_risk = df[df[target] < 20].copy()

print(f"\n高危岗位 (Risk > 80%): {len(high_risk)} 条 ({len(high_risk)/len(df)*100:.1f}%)")
print(f"低危岗位 (Risk < 20%): {len(low_risk)} 条 ({len(low_risk)/len(df)*100:.1f}%)")

print("\n--- 高危岗位 Top 15 岗位名称 ---")
print(high_risk['Job Title'].value_counts().head(15).to_string())

print("\n--- 低危岗位 Top 15 岗位名称 ---")
print(low_risk['Job Title'].value_counts().head(15).to_string())

# 图12: 高危 vs 低危岗位的特征对比雷达图
print("\n--- 高危 vs 低危岗位特征均值对比 ---")
compare_cols = ['Median Salary (USD)', 'Experience Required (Years)',
                'Remote Work Ratio (%)', 'Gender Diversity (%)']
for col in compare_cols:
    h = high_risk[col].mean()
    l = low_risk[col].mean()
    diff_pct = (h - l) / l * 100 if l != 0 else 0
    print(f"  {col:40s}  高危={h:10.2f}  低危={l:10.2f}  差异={diff_pct:+.2f}%")

# 图12: 高危vs低危岗位特征对比
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
compare_plot_cols = ['Median Salary (USD)', 'Experience Required (Years)',
                     'Remote Work Ratio (%)', 'Gender Diversity (%)']
for idx, col in enumerate(compare_plot_cols):
    ax = axes[idx // 2][idx % 2]
    compare_df = pd.DataFrame({
        'High Risk': high_risk[col],
        'Low Risk': low_risk[col]
    })
    sns.kdeplot(data=compare_df, ax=ax, fill=True, alpha=0.5)
    ax.set_title(f'{col}\n高危 vs 低危分布对比', fontsize=11)
    ax.legend(fontsize=10)
plt.suptitle('图12: 高危 vs 低危岗位数值特征分布对比', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig12_high_vs_low_risk_comparison.png'))
plt.close()
print(">>> 图12已保存: fig12_high_vs_low_risk_comparison.png")

# ============================================================
# 10. 岗位增长趋势与自动化风险
# ============================================================
print("\n" + "=" * 70)
print("  第十部分：岗位增长趋势与自动化风险")
print("=" * 70)

df['growth_rate'] = (df['Projected Openings (2030)'] - df['Job Openings (2024)']) / df['Job Openings (2024)']
df['growth_category'] = pd.cut(df['growth_rate'],
                                bins=[-np.inf, -0.3, 0, 0.3, 1, np.inf],
                                labels=['大幅缩减', '小幅缩减', '平稳', '小幅增长', '大幅增长'])

print("\n--- 岗位增长类别分布 ---")
print(df['growth_category'].value_counts().sort_index())

print("\n--- 各增长类别平均自动化风险 ---")
print(df.groupby('growth_category')[target].mean().round(2))

# 图13: 增长率 vs 自动化风险散点图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(df['growth_rate'], df[target], alpha=0.15, s=8, c='steelblue')
z = np.polyfit(df['growth_rate'], df[target], 1)
p = np.poly1d(z)
x_line = np.linspace(df['growth_rate'].min(), df['growth_rate'].max(), 100)
axes[0].plot(x_line, p(x_line), 'r--', linewidth=2, label=f'线性拟合 (slope={z[0]:.3f})')
axes[0].set_xlabel('Growth Rate (Projected 2030 vs 2024)', fontsize=12)
axes[0].set_ylabel('Automation Risk (%)', fontsize=12)
axes[0].set_title('岗位增长率 vs 自动化风险', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# 按增长类别画箱线图
sns.boxplot(data=df, x='growth_category', y=target, ax=axes[1], palette='RdYlGn')
axes[1].set_xlabel('Growth Category', fontsize=12)
axes[1].set_ylabel('Automation Risk (%)', fontsize=12)
axes[1].set_title('各增长类别自动化风险分布', fontsize=14)
axes[1].tick_params(axis='x', rotation=15)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('图13: 岗位增长趋势与自动化风险关系', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig13_growth_vs_risk.png'))
plt.close()
print(">>> 图13已保存: fig13_growth_vs_risk.png")

# 增长率与风险的相关性
corr_growth = df['growth_rate'].corr(df[target])
spearman_growth = df['growth_rate'].corr(df[target], method='spearman')
print(f"\n  增长率 vs 风险 Pearson r = {corr_growth:.4f}")
print(f"  增长率 vs 风险 Spearman ρ = {spearman_growth:.4f}")

# ============================================================
# 11. 异常值检测
# ============================================================
print("\n" + "=" * 70)
print("  第十一部分：异常值检测(IQR方法)")
print("=" * 70)

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    pct = n_outliers / len(df) * 100
    print(f"  {col:40s}  异常值: {n_outliers:5d} ({pct:.2f}%)  范围: [{lower:.1f}, {upper:.1f}]")

# ============================================================
# 12. EDA总结
# ============================================================
print("\n" + "=" * 70)
print("  EDA 总结与关键发现")
print("=" * 70)

print("""
  【关键发现汇总】

  1. 数据完整性: 30,000条记录 × 13个特征, 无缺失值, 数据质量良好

  2. 目标变量分布:
     - Automation Risk (%) 近似均匀分布 (均值≈50, 范围0-100)
     - 三分类: Low Risk ≈ 30%, Medium Risk ≈ 30%, High Risk ≈ 40%

  3. 线性相关极弱:
     - 所有数值特征与Automation Risk的Pearson r < 0.01
     - Spearman秩相关同样极弱
     - 结论: 单一特征无法线性预测风险, 必须依赖多维特征交互+非线性模型

  4. 分类特征差异:
     - 行业间风险均值差异小 (49.6-50.8), 但Kruskal-Wallis检验可能显著
     - AI Impact Level (Low/Moderate/High) 与风险关系需非线性建模验证

  5. 高危岗位特征(初步):
     - Risk>80%的岗位占20%, 分布广泛无明显集中
     - 高危与低危岗位在薪资、经验等维度差异较小
     - 需要通过多维度交叉才能有效识别高危岗位

  6. 岗位增长趋势:
     - 增长率与自动化风险线性相关极弱
     - 存在"增长但仍高危"和"缩减但低危"的结构性矛盾岗位

  7. 建模建议:
     - 必须进行特征交互(如: Industry×Education, Remote×Salary等)
     - 优先使用树模型(LightGBM/XGBoost)捕捉非线性关系
     - 聚类分析可辅助发现高危岗位群组
     - SHAP分析对结果可解释性至关重要
""")

print("=" * 70)
print("  EDA完成! 共生成13张分析图表, 保存在 outputs/ 目录")
print("=" * 70)
