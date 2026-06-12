# -*- coding: utf-8 -*-
"""
=================================================================
  基于多维度特征的劳动力AI替代风险挖掘与高危岗位识别
  Step 6: 关联规则挖掘 (成员C负责)
  方法: Apriori算法
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
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
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  第六步：关联规则挖掘")
print("=" * 70)

# ============================================================
# 1. 加载原始数据
# ============================================================
print("\n【1】加载数据...")

raw_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'ai_job_trends_dataset.csv'))
print(f"  数据形状: {raw_df.shape}")

# ============================================================
# 2. 构建事务数据
# ============================================================
print("\n【2】构建事务数据 (离散化)...")

df = raw_df.copy()

# 风险类别
df['Risk_Level'] = pd.cut(df['Automation Risk (%)'],
                           bins=[-0.01, 30, 60, 100],
                           labels=['Risk_Low', 'Risk_Medium', 'Risk_High'])

# 薪资等级
df['Salary_Level'] = pd.cut(df['Median Salary (USD)'],
                             bins=[0, 60000, 90000, 120000, 150000],
                             labels=['Salary_Low', 'Salary_Med', 'Salary_High', 'Salary_VHigh'])

# 经验等级
df['Exp_Level'] = pd.cut(df['Experience Required (Years)'],
                          bins=[-1, 2, 5, 10, 20, 100],
                          labels=['Exp_Entry', 'Exp_Junior', 'Exp_Mid', 'Exp_Senior', 'Exp_Expert'])

# 远程工作等级
df['Remote_Level'] = pd.cut(df['Remote Work Ratio (%)'],
                             bins=[-1, 25, 50, 75, 100],
                             labels=['Remote_Low', 'Remote_Med', 'Remote_High', 'Remote_VHigh'])

# 性别多样性等级
df['Diversity_Level'] = pd.cut(df['Gender Diversity (%)'],
                                bins=[0, 35, 50, 65, 100],
                                labels=['Diversity_Low', 'Diversity_Med', 'Diversity_High', 'Diversity_VHigh'])

# 岗位增长趋势
df['Growth_Trend'] = df['Job Status'].map({'Increasing': 'Growth_Inc', 'Decreasing': 'Growth_Dec'})

# AI冲击等级
df['AI_Impact'] = df['AI Impact Level'].map({'Low': 'AI_Low', 'Moderate': 'AI_Moderate', 'High': 'AI_High'})

# 构建事务列表
transaction_cols = ['Industry', 'Risk_Level', 'Salary_Level', 'Exp_Level',
                    'Remote_Level', 'Diversity_Level', 'Growth_Trend', 'AI_Impact',
                    'Required Education']

transactions = df[transaction_cols].astype(str).values.tolist()

print(f"  事务数量: {len(transactions)}")
print(f"  示例事务: {transactions[0]}")

# ============================================================
# 3. Apriori算法挖掘频繁项集
# ============================================================
print("\n【3】Apriori频繁项集挖掘...")

# 编码事务数据
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print(f"  编码后维度: {df_encoded.shape}")

# 挖掘频繁项集 (最小支持度=0.05)
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True, max_len=4)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)

print(f"\n  频繁项集数量: {len(frequent_itemsets)}")
print(f"  2项频繁项集数量: {len(frequent_itemsets[frequent_itemsets['length'] == 2])}")
print(f"  3项频繁项集数量: {len(frequent_itemsets[frequent_itemsets['length'] == 3])}")

# Top 20 频繁项集
print(f"\n--- Top 20 频繁项集 (按支持度排序) ---")
frequent_sorted = frequent_itemsets.sort_values('support', ascending=False).head(20)
for _, row in frequent_sorted.iterrows():
    items = ', '.join(sorted(row['itemsets']))
    print(f"  Support={row['support']:.4f}  Length={row['length']}  Items: {items}")

# 保存频繁项集
frequent_itemsets.to_csv(os.path.join(OUTPUT_DIR, 'frequent_itemsets.csv'), index=False)
print(f"\n  ✓ 频繁项集已保存: frequent_itemsets.csv")

# ============================================================
# 4. 生成关联规则
# ============================================================
print("\n【4】关联规则生成...")

rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.3)
rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))

print(f"  关联规则数量: {len(rules)}")

# 筛选与高危风险相关的规则
high_risk_rules = rules[
    rules['consequents'].apply(lambda x: 'Risk_High' in x) |
    rules['antecedents'].apply(lambda x: 'Risk_High' in x)
]

print(f"  与高危风险相关的规则数量: {len(high_risk_rules)}")

# 按提升度排序
high_risk_rules_sorted = high_risk_rules.sort_values('lift', ascending=False)

print(f"\n--- 高危风险关联规则 Top 15 (按提升度) ---")
for i, (_, row) in enumerate(high_risk_rules_sorted.head(15).iterrows()):
    print(f"  {i+1:2d}. {row['antecedents_str']:40s} → {row['consequents_str']:15s}  "
          f"Support={row['support']:.4f}  Confidence={row['confidence']:.4f}  Lift={row['lift']:.4f}")

# 保存关联规则
rules.to_csv(os.path.join(OUTPUT_DIR, 'all_association_rules.csv'), index=False)
high_risk_rules_sorted.to_csv(os.path.join(OUTPUT_DIR, 'high_risk_association_rules.csv'), index=False)
print(f"\n  ✓ 全部规则已保存: all_association_rules.csv")
print(f"  ✓ 高危规则已保存: high_risk_association_rules.csv")

# ============================================================
# 5. 关联规则可视化
# ============================================================
print("\n【5】关联规则可视化...")

# 图30: 关联规则散点图 (Support vs Confidence)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 30a: 全部规则
scatter = axes[0].scatter(rules['support'], rules['confidence'],
                          c=rules['lift'], cmap='RdYlGn', alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
plt.colorbar(scatter, ax=axes[0], label='Lift')
axes[0].set_xlabel('Support', fontsize=12)
axes[0].set_ylabel('Confidence', fontsize=12)
axes[0].set_title('关联规则: Support vs Confidence', fontsize=14)
axes[0].grid(alpha=0.3)

# 30b: 高危规则
if len(high_risk_rules_sorted) > 0:
    top_rules = high_risk_rules_sorted.head(20)
    scatter2 = axes[1].scatter(top_rules['support'], top_rules['confidence'],
                               c=top_rules['lift'], cmap='Reds', alpha=0.8, s=60, edgecolors='white')
    plt.colorbar(scatter2, ax=axes[1], label='Lift')
    # 标注规则
    for i, (_, row) in enumerate(top_rules.iterrows()):
        if i < 8:
            axes[1].annotate(f"R{i+1}", (row['support'], row['confidence']),
                            fontsize=8, ha='center')
    axes[1].set_xlabel('Support', fontsize=12)
    axes[1].set_ylabel('Confidence', fontsize=12)
    axes[1].set_title('高危风险关联规则 Top 20', fontsize=14)
    axes[1].grid(alpha=0.3)

plt.suptitle('图30: 关联规则散点图', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig30_association_rules_scatter.png'))
plt.close()
print(">>> 图30已保存: fig30_association_rules_scatter.png")

# 图31: 高危规则网络图 (简化版)
print("\n【5.1】关联规则热力图...")

# 只看2项规则 (前件1项→后件1项)
two_item_rules = high_risk_rules_sorted[
    (high_risk_rules_sorted['antecedents'].apply(len) == 1) &
    (high_risk_rules_sorted['consequents'].apply(len) == 1)
].copy()

if len(two_item_rules) > 0:
    # 构建pivot表
    two_item_rules['antecedent'] = two_item_rules['antecedents'].apply(lambda x: list(x)[0])
    two_item_rules['consequent'] = two_item_rules['consequents'].apply(lambda x: list(x)[0])

    pivot = two_item_rules.pivot_table(
        index='antecedent', columns='consequent', values='lift', aggfunc='mean'
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, linewidths=0.5)
    ax.set_title('图31: 关联规则提升度热力图 (前件→后件)', fontsize=14)
    ax.set_xlabel('Consequent (后件)', fontsize=12)
    ax.set_ylabel('Antecedent (前件)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig31_association_lift_heatmap.png'))
    plt.close()
    print(">>> 图31已保存: fig31_association_lift_heatmap.png")

# ============================================================
# 6. 关键发现总结
# ============================================================
print("\n【6】关联规则关键发现...")

# 找出提升度最高的规则
print("\n--- 提升度最高的10条规则 ---")
for i, (_, row) in enumerate(rules.sort_values('lift', ascending=False).head(10).iterrows()):
    print(f"  {i+1}. {row['antecedents_str']:40s} → {row['consequents_str']:15s}  "
          f"Sup={row['support']:.4f}  Conf={row['confidence']:.4f}  Lift={row['lift']:.4f}")

# 分析哪些特征组合最容易导致高危
print("\n--- 高危岗位的关键特征组合 ---")
# 筛选: 前件不含Risk, 后件含Risk_High
predictive_rules = rules[
    (~rules['antecedents'].apply(lambda x: any('Risk' in i for i in x))) &
    (rules['consequents'].apply(lambda x: 'Risk_High' in x))
].sort_values('lift', ascending=False)

if len(predictive_rules) > 0:
    for i, (_, row) in enumerate(predictive_rules.head(10).iterrows()):
        print(f"  {i+1}. {row['antecedents_str']:50s} → Risk_High  "
              f"Sup={row['support']:.4f}  Conf={row['confidence']:.4f}  Lift={row['lift']:.4f}")

# ============================================================
# 7. 总结
# ============================================================
print("\n" + "=" * 70)
print("  关联规则挖掘完成!")
print("=" * 70)

print(f"""
  【关联规则总结】

  1. 频繁项集:
     - 总数: {len(frequent_itemsets)}
     - 2项频繁集: {len(frequent_itemsets[frequent_itemsets['length'] == 2])}
     - 3项频繁集: {len(frequent_itemsets[frequent_itemsets['length'] == 3])}

  2. 关联规则:
     - 总数: {len(rules)}
     - 与高危相关: {len(high_risk_rules)}

  3. 生成文件:
     - outputs/frequent_itemsets.csv
     - outputs/all_association_rules.csv
     - outputs/high_risk_association_rules.csv
     - outputs/fig30-31 共2张图
""")
