# -*- coding: utf-8 -*-
"""
=================================================================
  基于多维度特征的劳动力AI替代风险挖掘与高危岗位识别
  Step 2: 数据准备与特征工程
  负责人: 成员A
=================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'ai_job_trends_dataset.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

print("=" * 70)
print("  数据准备与特征工程")
print("=" * 70)

# ============================================================
# 1. 数据加载
# ============================================================
print("\n【1】加载原始数据...")
df = pd.read_csv(DATA_PATH)
print(f"  原始数据形状: {df.shape}")
print(f"  列名: {list(df.columns)}")

# ============================================================
# 2. 基础特征工程
# ============================================================
print("\n【2】构建交互特征...")

# (a) 岗位增长率
df['growth_rate'] = (df['Projected Openings (2030)'] - df['Job Openings (2024)']) / df['Job Openings (2024)']
print(f"  ✓ 特征: growth_rate (岗位增长率)")

# (b) 单位经验薪资 (薪资/经验)
df['salary_per_experience'] = df['Median Salary (USD)'] / (df['Experience Required (Years)'] + 1)
print(f"  ✓ 特征: salary_per_experience (单位经验薪资)")

# (c) 岗位密度 (岗位数/经验要求)
df['job_density'] = df['Job Openings (2024)'] / (df['Experience Required (Years)'] + 1)
print(f"  ✓ 特征: job_density (岗位密度)")

# (d) 薪资与远程工作交互
df['salary_remote_interaction'] = df['Median Salary (USD)'] * df['Remote Work Ratio (%)'] / 100
print(f"  ✓ 特征: salary_remote_interaction (薪资×远程)")

# (e) 岗位增长×AI冲击等级交互
df['growth_impact_interaction'] = df['growth_rate'] * df['AI Impact Level'].map({'Low': 1, 'Moderate': 2, 'High': 3})
print(f"  ✓ 特征: growth_impact_interaction (增长×AI冲击)")

# (f) 薪资与经验交互
df['salary_experience_interaction'] = df['Median Salary (USD)'] * df['Experience Required (Years)']
print(f"  ✓ 特征: salary_experience_interaction (薪资×经验)")

# (g) 性别多样性与岗位增长交互
df['diversity_growth_interaction'] = df['Gender Diversity (%)'] * df['growth_rate']
print(f"  ✓ 特征: diversity_growth_interaction (性别多样性×增长)")

# (h) 远程工作与岗位增长交互
df['remote_growth_interaction'] = df['Remote Work Ratio (%)'] * df['growth_rate']
print(f"  ✓ 特征: remote_growth_interaction (远程×增长)")

# (i) 薪资标准化 (Z-score)
df['salary_zscore'] = (df['Median Salary (USD)'] - df['Median Salary (USD)'].mean()) / df['Median Salary (USD)'].std()
print(f"  ✓ 特征: salary_zscore (薪资标准化)")

# (j) 经验等级分类
df['experience_level'] = pd.cut(df['Experience Required (Years)'],
                                 bins=[-1, 2, 5, 10, 20, 100],
                                 labels=['Entry', 'Junior', 'Mid', 'Senior', 'Expert'])
print(f"  ✓ 特征: experience_level (经验等级)")

# (k) 薪资等级分类
df['salary_level'] = pd.cut(df['Median Salary (USD)'],
                             bins=[0, 60000, 90000, 120000, 150000],
                             labels=['Low', 'Medium', 'High', 'VeryHigh'])
print(f"  ✓ 特征: salary_level (薪资等级)")

# (l) 远程工作等级
df['remote_level'] = pd.cut(df['Remote Work Ratio (%)'],
                             bins=[-1, 25, 50, 75, 100],
                             labels=['Low', 'Medium', 'High', 'VeryHigh'])
print(f"  ✓ 特征: remote_level (远程等级)")

# (m) 岗位增长分类
df['growth_category'] = pd.cut(df['growth_rate'],
                                bins=[-np.inf, -0.3, 0, 0.3, 1, np.inf],
                                labels=['SevereDecline', 'ModerateDecline', 'Stable', 'ModerateGrowth', 'HighGrowth'])
print(f"  ✓ 特征: growth_category (增长分类)")

print(f"\n  特征工程后数据形状: {df.shape}")

# ============================================================
# 3. 有序编码
# ============================================================
print("\n【3】有序特征编码...")

# 学历有序编码 (High School < Associate < Bachelor < Master < PhD)
education_order = {'High School': 0, 'Associate Degree': 1, "Bachelor's Degree": 2, "Master's Degree": 3, 'PhD': 4}
df['education_encoded'] = df['Required Education'].map(education_order)
print(f"  ✓ education_encoded: {education_order}")

# AI Impact Level有序编码
df['impact_encoded'] = df['AI Impact Level'].map({'Low': 0, 'Moderate': 1, 'High': 2})
print(f"  ✓ impact_encoded: Low=0, Moderate=1, High=2")

# Job Status编码
df['status_encoded'] = df['Job Status'].map({'Decreasing': 0, 'Increasing': 1})
print(f"  ✓ status_encoded: Decreasing=0, Increasing=1")

# 经验等级编码
exp_level_order = {'Entry': 0, 'Junior': 1, 'Mid': 2, 'Senior': 3, 'Expert': 4}
df['experience_level_encoded'] = df['experience_level'].map(exp_level_order)
print(f"  ✓ experience_level_encoded: {exp_level_order}")

# 薪资等级编码
salary_level_order = {'Low': 0, 'Medium': 1, 'High': 2, 'VeryHigh': 3}
df['salary_level_encoded'] = df['salary_level'].map(salary_level_order)
print(f"  ✓ salary_level_encoded: {salary_level_order}")

# 远程等级编码
remote_level_order = {'Low': 0, 'Medium': 1, 'High': 2, 'VeryHigh': 3}
df['remote_level_encoded'] = df['remote_level'].map(remote_level_order)
print(f"  ✓ remote_level_encoded: {remote_level_order}")

# 增长分类编码
growth_cat_order = {'SevereDecline': 0, 'ModerateDecline': 1, 'Stable': 2, 'ModerateGrowth': 3, 'HighGrowth': 4}
df['growth_category_encoded'] = df['growth_category'].map(growth_cat_order)
print(f"  ✓ growth_category_encoded: {growth_cat_order}")

# ============================================================
# 4. 目标编码 (Job Title - 高基数类别)
# ============================================================
print("\n【4】Job Title目标编码...")
print(f"  Job Title类别数: {df['Job Title'].nunique()}")

# 使用TargetEncoder进行目标编码
te = TargetEncoder()
df['job_title_encoded'] = te.fit_transform(df['Job Title'], df['Automation Risk (%)'])
print(f"  ✓ job_title_encoded 完成")

# 保存encoder供后续使用
with open(os.path.join(OUTPUT_DIR, 'target_encoder.pkl'), 'wb') as f:
    pickle.dump(te, f)
print(f"  ✓ 目标编码器已保存: target_encoder.pkl")

# ============================================================
# 5. One-Hot编码 (低基数类别)
# ============================================================
print("\n【5】One-Hot编码...")

# 需要One-Hot编码的列
onehot_cols = ['Industry', 'Location']

# 执行One-Hot编码
df_encoded = pd.get_dummies(df, columns=onehot_cols, drop_first=True, prefix=onehot_cols)
print(f"  ✓ One-Hot编码完成: {onehot_cols}")

# 查看生成的列
onehot_generated = [c for c in df_encoded.columns if any(c.startswith(col + '_') for col in onehot_cols)]
print(f"  生成One-Hot列数: {len(onehot_generated)}")

# ============================================================
# 6. 目标变量处理
# ============================================================
print("\n【6】目标变量处理...")

# 回归目标: 原始Automation Risk (%)
y_reg = df_encoded['Automation Risk (%)'].copy()
print(f"  ✓ 回归目标: Automation Risk (%), 范围 [{y_reg.min():.2f}, {y_reg.max():.2f}]")

# 分类目标: 三分类
# 方法1: 等宽分箱 (0-30, 30-60, 60-100)
df_encoded['risk_category_equal'] = pd.cut(df_encoded['Automation Risk (%)'],
                                            bins=[-0.01, 30, 60, 100],
                                            labels=[0, 1, 2])
print(f"  ✓ 分类目标1 (等宽分箱): Low=0, Medium=1, High=2")
print(f"      Low Risk: {(df_encoded['risk_category_equal']==0).sum()}")
print(f"      Medium Risk: {(df_encoded['risk_category_equal']==1).sum()}")
print(f"      High Risk: {(df_encoded['risk_category_equal']==2).sum()}")

# 方法2: 三分位数分箱
df_encoded['risk_category_quantile'] = pd.qcut(df_encoded['Automation Risk (%)'],
                                                q=3,
                                                labels=[0, 1, 2])
print(f"  ✓ 分类目标2 (三分位数): Low=0, Medium=1, High=2")
print(f"      Low Risk: {(df_encoded['risk_category_quantile']==0).sum()}")
print(f"      Medium Risk: {(df_encoded['risk_category_quantile']==1).sum()}")
print(f"      High Risk: {(df_encoded['risk_category_quantile']==2).sum()}")

y_cls_equal = df_encoded['risk_category_equal'].copy()
y_cls_quantile = df_encoded['risk_category_quantile'].copy()

# ============================================================
# 7. 特征选择
# ============================================================
print("\n【7】特征选择...")

# 排除不需要的列
exclude_cols = [
    'Job Title', 'Required Education', 'AI Impact Level', 'Job Status',
    'experience_level', 'salary_level', 'remote_level', 'growth_category',
    'Automation Risk (%)', 'risk_category_equal', 'risk_category_quantile'
]

# 选择特征列
feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]
print(f"  最终特征数量: {len(feature_cols)}")
print(f"  特征列表:")
for i, col in enumerate(feature_cols, 1):
    print(f"    {i:2d}. {col}")

X = df_encoded[feature_cols].copy()

# ============================================================
# 8. 数据划分
# ============================================================
print("\n【8】数据划分...")

# 回归任务划分
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cls_equal)

# 分类任务划分 (等宽分箱)
X_train_cls_eq, X_test_cls_eq, y_train_cls_eq, y_test_cls_eq = train_test_split(
    X, y_cls_equal, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cls_equal)

# 分类任务划分 (三分位数分箱)
X_train_cls_qt, X_test_cls_qt, y_train_cls_qt, y_test_cls_qt = train_test_split(
    X, y_cls_quantile, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cls_quantile)

print(f"  ✓ 训练集大小: {len(X_train_reg)} ({len(X_train_reg)/len(X)*100:.1f}%)")
print(f"  ✓ 测试集大小: {len(X_test_reg)} ({len(X_test_reg)/len(X)*100:.1f}%)")

# ============================================================
# 9. 标准化 (可选 - 对树模型不需要, 但对SVM/神经网络需要)
# ============================================================
print("\n【9】数值特征标准化...")

# 识别数值列 (非One-Hot列)
numeric_feature_cols = [c for c in feature_cols if not any(c.startswith(prefix + '_') for prefix in onehot_cols)]
print(f"  需要标准化的数值特征: {len(numeric_feature_cols)} 个")

scaler = StandardScaler()
X_train_scaled = X_train_reg.copy()
X_test_scaled = X_test_reg.copy()

X_train_scaled[numeric_feature_cols] = scaler.fit_transform(X_train_reg[numeric_feature_cols])
X_test_scaled[numeric_feature_cols] = scaler.transform(X_test_reg[numeric_feature_cols])

# 保存scaler
with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ 标准化器已保存: scaler.pkl")

# ============================================================
# 10. 保存处理后的数据
# ============================================================
print("\n【10】保存处理后的数据...")

# 保存完整特征数据
X.to_csv(os.path.join(OUTPUT_DIR, 'X_full.csv'), index=False)
y_reg.to_csv(os.path.join(OUTPUT_DIR, 'y_regression.csv'), index=False)
y_cls_equal.to_csv(os.path.join(OUTPUT_DIR, 'y_classification_equal.csv'), index=False)
y_cls_quantile.to_csv(os.path.join(OUTPUT_DIR, 'y_classification_quantile.csv'), index=False)
print(f"  ✓ X_full.csv 保存完成")
print(f"  ✓ y_regression.csv 保存完成")
print(f"  ✓ y_classification_equal.csv 保存完成")
print(f"  ✓ y_classification_quantile.csv 保存完成")

# 保存训练/测试集 (未标准化 - 用于树模型)
X_train_reg.to_csv(os.path.join(OUTPUT_DIR, 'X_train_reg.csv'), index=False)
X_test_reg.to_csv(os.path.join(OUTPUT_DIR, 'X_test_reg.csv'), index=False)
y_train_reg.to_csv(os.path.join(OUTPUT_DIR, 'y_train_reg.csv'), index=False)
y_test_reg.to_csv(os.path.join(OUTPUT_DIR, 'y_test_reg.csv'), index=False)

X_train_cls_eq.to_csv(os.path.join(OUTPUT_DIR, 'X_train_cls_eq.csv'), index=False)
X_test_cls_eq.to_csv(os.path.join(OUTPUT_DIR, 'X_test_cls_eq.csv'), index=False)
y_train_cls_eq.to_csv(os.path.join(OUTPUT_DIR, 'y_train_cls_eq.csv'), index=False)
y_test_cls_eq.to_csv(os.path.join(OUTPUT_DIR, 'y_test_cls_eq.csv'), index=False)
print(f"  ✓ 分类数据(等宽分箱)保存完成")

# 保存标准化后的数据 (用于SVM/神经网络)
X_train_scaled.to_csv(os.path.join(OUTPUT_DIR, 'X_train_scaled.csv'), index=False)
X_test_scaled.to_csv(os.path.join(OUTPUT_DIR, 'X_test_scaled.csv'), index=False)
print(f"  ✓ 标准化数据保存完成")

# 保存特征列名
with open(os.path.join(OUTPUT_DIR, 'feature_columns.pkl'), 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"  ✓ 特征列名已保存: feature_columns.pkl")

# ============================================================
# 11. 数据质量检查
# ============================================================
print("\n【11】数据质量检查...")

print(f"\n  训练集特征统计:")
print(f"    - 形状: {X_train_reg.shape}")
print(f"    - 缺失值: {X_train_reg.isnull().sum().sum()}")
print(f"    - 无穷值: {np.isinf(X_train_reg.select_dtypes(include=[np.number])).sum().sum()}")

print(f"\n  测试集特征统计:")
print(f"    - 形状: {X_test_reg.shape}")
print(f"    - 缺失值: {X_test_reg.isnull().sum().sum()}")
print(f"    - 无穷值: {np.isinf(X_test_reg.select_dtypes(include=[np.number])).sum().sum()}")

# 检查类别分布
print(f"\n  分类目标分布 (训练集 - 等宽分箱):")
print(y_train_cls_eq.value_counts().sort_index())

# ============================================================
# 12. 特征重要性初探 (基于单变量统计)
# ============================================================
print("\n【12】特征重要性初探...")

from sklearn.feature_selection import mutual_info_classif

# 填充缺失值 (One-Hot编码产生的NaN用0填充)
X_train_filled = X_train_cls_eq.fillna(0)
X_test_filled = X_test_cls_eq.fillna(0)

# 使用互信息评估特征重要性
mi_scores = mutual_info_classif(X_train_filled, y_train_cls_eq, random_state=RANDOM_STATE)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mutual_info': mi_scores
}).sort_values('mutual_info', ascending=False)

print(f"\n  互信息评分 Top 10:")
for i, row in mi_df.head(10).iterrows():
    print(f"    {row['feature']:40s}  MI = {row['mutual_info']:.4f}")

# 保存互信息评分
mi_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_mutual_info.csv'), index=False)
print(f"\n  ✓ 特征重要性评分已保存: feature_importance_mutual_info.csv")

# 同时保存填充后的数据
X_train_filled.to_csv(os.path.join(OUTPUT_DIR, 'X_train_filled.csv'), index=False)
X_test_filled.to_csv(os.path.join(OUTPUT_DIR, 'X_test_filled.csv'), index=False)
print(f"  ✓ 填充后数据已保存")

# ============================================================
# 13. 生成数据字典
# ============================================================
print("\n【13】生成数据字典...")

data_dict = {
    '原始特征': {
        'Job Title': '岗位名称 (639类, 目标编码)',
        'Industry': '行业 (8类, One-Hot编码)',
        'Job Status': '岗位趋势 (Increasing/Decreasing)',
        'AI Impact Level': 'AI冲击等级 (Low/Moderate/High)',
        'Median Salary (USD)': '中位数薪资',
        'Required Education': '学历要求 (5类)',
        'Experience Required (Years)': '经验要求年限',
        'Job Openings (2024)': '2024年岗位开放数',
        'Projected Openings (2030)': '2030年预测岗位数',
        'Remote Work Ratio (%)': '远程工作比例',
        'Automation Risk (%)': '自动化替代风险 (目标变量)',
        'Location': '地区 (8类, One-Hot编码)',
        'Gender Diversity (%)': '性别多样性比例'
    },
    '衍生特征': {
        'growth_rate': '岗位增长率 = (2030-2024)/2024',
        'salary_per_experience': '单位经验薪资 = Salary/(Exp+1)',
        'job_density': '岗位密度 = Openings/(Exp+1)',
        'salary_remote_interaction': '薪资×远程交互',
        'growth_impact_interaction': '增长×AI冲击交互',
        'salary_experience_interaction': '薪资×经验交互',
        'diversity_growth_interaction': '性别多样性×增长交互',
        'remote_growth_interaction': '远程×增长交互',
        'salary_zscore': '薪资标准化Z-score',
        'experience_level': '经验等级 (Entry/Junior/Mid/Senior/Expert)',
        'salary_level': '薪资等级 (Low/Medium/High/VeryHigh)',
        'remote_level': '远程等级 (Low/Medium/High/VeryHigh)',
        'growth_category': '增长分类 (5类)',
        'job_title_encoded': '岗位名称目标编码'
    },
    '编码特征': {
        'education_encoded': '学历有序编码 (0-4)',
        'impact_encoded': 'AI冲击有序编码 (0-2)',
        'status_encoded': '岗位趋势编码 (0-1)',
        'experience_level_encoded': '经验等级编码 (0-4)',
        'salary_level_encoded': '薪资等级编码 (0-3)',
        'remote_level_encoded': '远程等级编码 (0-3)',
        'growth_category_encoded': '增长分类编码 (0-4)'
    },
    '目标变量': {
        'y_regression': 'Automation Risk (%) 连续值 [0,100]',
        'y_classification_equal': '三分类 (等宽分箱: 0-30, 30-60, 60-100)',
        'y_classification_quantile': '三分类 (三分位数分箱)'
    }
}

# 保存数据字典
with open(os.path.join(OUTPUT_DIR, 'data_dictionary.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("  数据字典\n")
    f.write("=" * 70 + "\n\n")
    for section, items in data_dict.items():
        f.write(f"【{section}】\n")
        for key, value in items.items():
            f.write(f"  {key:35s} : {value}\n")
        f.write("\n")

print(f"  ✓ 数据字典已保存: data_dictionary.txt")

# ============================================================
# 14. 总结
# ============================================================
print("\n" + "=" * 70)
print("  数据准备完成!")
print("=" * 70)

print(f"""
  【输出文件清单】

  1. 特征数据:
     - X_full.csv                    : 完整特征矩阵 ({X.shape[0]} × {X.shape[1]})
     - X_train_reg.csv / X_test_reg.csv : 回归任务训练/测试集
     - X_train_cls_eq.csv / X_test_cls_eq.csv : 分类任务训练/测试集
     - X_train_scaled.csv / X_test_scaled.csv : 标准化后数据

  2. 目标变量:
     - y_regression.csv              : 回归目标 (连续值)
     - y_classification_equal.csv    : 分类目标 (等宽分箱)
     - y_classification_quantile.csv : 分类目标 (三分位数分箱)

  3. 辅助文件:
     - target_encoder.pkl            : Job Title目标编码器
     - scaler.pkl                    : 标准化器
     - feature_columns.pkl           : 特征列名列表
     - feature_importance_mutual_info.csv : 互信息特征重要性
     - data_dictionary.txt           : 数据字典

  【关键统计】
  - 原始特征: 13 个
  - 衍生特征: 13 个
  - 编码特征: 7 个
  - One-Hot特征: {len(onehot_generated)} 个
  - 总计特征: {len(feature_cols)} 个
  - 训练样本: {len(X_train_reg)} 条
  - 测试样本: {len(X_test_reg)} 条
""")

print("=" * 70)
print("  下一步: 成员B可进行分类建模, 成员C可进行聚类分析")
print("=" * 70)
