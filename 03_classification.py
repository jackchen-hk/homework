# -*- coding: utf-8 -*-
"""
=================================================================
  基于多维度特征的劳动力AI替代风险挖掘与高危岗位识别
  Step 3: 分类建模 (成员B负责)
  模型: LightGBM, XGBoost, Random Forest, SVM
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, GridSearchCV
import lightgbm as lgb
import xgboost as xgb
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
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

# 类别标签
CLASS_NAMES = ['Low Risk (0-30%)', 'Medium Risk (30-60%)', 'High Risk (60-100%)']

print("=" * 70)
print("  第三步：分类建模")
print("=" * 70)

# ============================================================
# 1. 加载处理后的数据
# ============================================================
print("\n【1】加载处理后的数据...")

X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train_filled.csv'))
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test_filled.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train_cls_eq.csv')).squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test_cls_eq.csv')).squeeze()

with open(os.path.join(DATA_DIR, 'feature_columns.pkl'), 'rb') as f:
    feature_cols = pickle.load(f)

print(f"  训练集: {X_train.shape}")
print(f"  测试集: {X_test.shape}")
print(f"  特征数: {len(feature_cols)}")
print(f"  类别分布 (训练集): {dict(y_train.value_counts().sort_index())}")

# ============================================================
# 2. LightGBM 分类
# ============================================================
print("\n【2】LightGBM 分类模型...")

lgb_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_STATE,
    'verbose': -1,
    'n_jobs': -1
}

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_valid],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

lgb_pred = lgb_model.predict(X_test)
lgb_pred_class = np.argmax(lgb_pred, axis=1)
lgb_acc = accuracy_score(y_test, lgb_pred_class)
lgb_f1 = f1_score(y_test, lgb_pred_class, average='macro')

print(f"  LightGBM Accuracy: {lgb_acc:.4f}")
print(f"  LightGBM Macro-F1: {lgb_f1:.4f}")

# 保存模型
try:
    lgb_model.save_model(os.path.join(MODEL_DIR, 'lightgbm_model.txt'))
    print(f"  ✓ 模型已保存: lightgbm_model.txt")
except Exception as e:
    print(f"  ⚠ 模型保存失败: {e}")

# ============================================================
# 3. XGBoost 分类
# ============================================================
print("\n【3】XGBoost 分类模型...")

xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    learning_rate=0.05,
    max_depth=8,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric='mlogloss'
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

xgb_pred_proba = xgb_model.predict_proba(X_test)
xgb_pred_class = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred_class)
xgb_f1 = f1_score(y_test, xgb_pred_class, average='macro')

print(f"  XGBoost Accuracy: {xgb_acc:.4f}")
print(f"  XGBoost Macro-F1: {xgb_f1:.4f}")

# 保存模型
try:
    xgb_model.save_model(os.path.join(MODEL_DIR, 'xgboost_model.json'))
    print(f"  ✓ 模型已保存: xgboost_model.json")
except Exception as e:
    print(f"  ⚠ 模型保存失败: {e}")

# ============================================================
# 4. Random Forest 分类
# ============================================================
print("\n【4】Random Forest 分类模型...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred_class = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)
rf_acc = accuracy_score(y_test, rf_pred_class)
rf_f1 = f1_score(y_test, rf_pred_class, average='macro')

print(f"  Random Forest Accuracy: {rf_acc:.4f}")
print(f"  Random Forest Macro-F1: {rf_f1:.4f}")

# 保存模型
with open(os.path.join(MODEL_DIR, 'rf_model.pkl'), 'wb') as f:
    pickle.dump(rf_model, f)
print(f"  ✓ 模型已保存: rf_model.pkl")

# ============================================================
# 5. 模型对比
# ============================================================
print("\n【5】模型对比汇总...")

# 计算每个模型的详细指标
def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """计算分类模型的全面评估指标"""
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Macro Precision': precision_score(y_true, y_pred, average='macro'),
        'Macro Recall': recall_score(y_true, y_pred, average='macro'),
        'Macro F1': f1_score(y_true, y_pred, average='macro'),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted'),
        'ROC-AUC (OvR)': roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    }
    return metrics

results = []
results.append(evaluate_model(y_test, lgb_pred_class, lgb_pred, 'LightGBM'))
results.append(evaluate_model(y_test, xgb_pred_class, xgb_pred_proba, 'XGBoost'))
results.append(evaluate_model(y_test, rf_pred_class, rf_pred_proba, 'Random Forest'))

results_df = pd.DataFrame(results).set_index('Model')
print(results_df.round(4).to_string())

# 保存对比结果
results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'))

# 图14: 模型对比柱状图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy对比
ax = axes[0]
bars = ax.bar(results_df.index, results_df['Accuracy'], color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='white')
for bar, val in zip(bars, results_df['Accuracy']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('模型准确率对比', fontsize=14)
ax.set_ylim(0, max(results_df['Accuracy']) * 1.15)
ax.grid(axis='y', alpha=0.3)

# Macro-F1对比
ax = axes[1]
bars = ax.bar(results_df.index, results_df['Macro F1'], color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='white')
for bar, val in zip(bars, results_df['Macro F1']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylabel('Macro F1-Score', fontsize=12)
ax.set_title('模型Macro-F1对比', fontsize=14)
ax.set_ylim(0, max(results_df['Macro F1']) * 1.15)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('图14: 分类模型性能对比', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig14_model_comparison.png'))
plt.close()
print(">>> 图14已保存: fig14_model_comparison.png")

# ============================================================
# 6. 最佳模型详细评估
# ============================================================
print("\n【6】最佳模型详细评估...")

# 选择F1最高的模型
best_idx = results_df['Macro F1'].idxmax()
print(f"  最佳模型: {best_idx} (Macro F1 = {results_df.loc[best_idx, 'Macro F1']:.4f})")

if best_idx == 'LightGBM':
    best_pred = lgb_pred_class
    best_proba = lgb_pred
elif best_idx == 'XGBoost':
    best_pred = xgb_pred_class
    best_proba = xgb_pred_proba
else:
    best_pred = rf_pred_class
    best_proba = rf_pred_proba

# 分类报告
print(f"\n--- {best_idx} 分类报告 ---")
print(classification_report(y_test, best_pred, target_names=CLASS_NAMES))

# 图15: 混淆矩阵
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 15a: 混淆矩阵 (计数)
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_title(f'{best_idx} 混淆矩阵 (计数)', fontsize=14)

# 15b: 混淆矩阵 (归一化)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_title(f'{best_idx} 混淆矩阵 (比例)', fontsize=14)

plt.suptitle('图15: 最佳模型混淆矩阵', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig15_confusion_matrix.png'))
plt.close()
print(">>> 图15已保存: fig15_confusion_matrix.png")

# 图16: ROC曲线 (One-vs-Rest)
fig, ax = plt.subplots(figsize=(10, 8))
colors_roc = ['#2ecc71', '#f39c12', '#e74c3c']

for i, (name, color) in enumerate(zip(CLASS_NAMES, colors_roc)):
    y_true_bin = (y_test == i).astype(int)
    y_score = best_proba[:, i] if len(best_proba.shape) > 1 else best_proba[:, i]
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    auc_val = roc_auc_score(y_true_bin, y_score)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc_val:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'图16: {best_idx} ROC曲线 (One-vs-Rest)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig16_roc_curve.png'))
plt.close()
print(">>> 图16已保存: fig16_roc_curve.png")

# ============================================================
# 7. 特征重要性分析
# ============================================================
print("\n【7】特征重要性分析...")

# LightGBM 特征重要性
lgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance_split': lgb_model.feature_importance(importance_type='split'),
    'importance_gain': lgb_model.feature_importance(importance_type='gain')
})

lgb_importance_split = lgb_importance.sort_values('importance_split', ascending=False).head(15)
lgb_importance_gain = lgb_importance.sort_values('importance_gain', ascending=False).head(15)

print(f"\n--- LightGBM Top 15 特征 (split) ---")
for _, row in lgb_importance_split.iterrows():
    print(f"  {row['feature']:40s}  split={row['importance_split']:6d}")

print(f"\n--- LightGBM Top 15 特征 (gain) ---")
for _, row in lgb_importance_gain.iterrows():
    print(f"  {row['feature']:40s}  gain={row['importance_gain']:12.2f}")

# 保存特征重要性
lgb_importance.to_csv(os.path.join(OUTPUT_DIR, 'lgb_feature_importance.csv'), index=False)

# 图17: 特征重要性可视化
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 17a: Split重要性
ax = axes[0]
lgb_importance_split_sorted = lgb_importance_split.sort_values('importance_split')
ax.barh(lgb_importance_split_sorted['feature'], lgb_importance_split_sorted['importance_split'],
        color='steelblue', alpha=0.8, edgecolor='white')
ax.set_xlabel('Split Count', fontsize=12)
ax.set_title('LightGBM 特征重要性 (Split)', fontsize=14)

# 17b: Gain重要性
ax = axes[1]
lgb_importance_gain_sorted = lgb_importance_gain.sort_values('importance_gain')
ax.barh(lgb_importance_gain_sorted['feature'], lgb_importance_gain_sorted['importance_gain'],
        color='coral', alpha=0.8, edgecolor='white')
ax.set_xlabel('Gain', fontsize=12)
ax.set_title('LightGBM 特征重要性 (Gain)', fontsize=14)

plt.suptitle('图17: LightGBM 特征重要性 Top 15', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig17_feature_importance.png'))
plt.close()
print(">>> 图17已保存: fig17_feature_importance.png")

# ============================================================
# 8. LightGBM 超参调优 (可选)
# ============================================================
print("\n【8】LightGBM 超参调优...")

# 定义参数搜索空间 (精简版, 减少运行时间)
param_grid = {
    'num_leaves': [31, 63],
    'max_depth': [6, 8],
    'learning_rate': [0.05, 0.1],
    'min_child_samples': [20, 50]
}

lgb_clf = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    verbose=-1,
    n_jobs=1
)

grid_search = GridSearchCV(
    lgb_clf,
    param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=1,
    verbose=0
)

grid_search.fit(X_train, y_train)

print(f"  最佳参数: {grid_search.best_params_}")
print(f"  最佳CV Macro-F1: {grid_search.best_score_:.4f}")

# 用最佳参数在测试集评估
best_lgb = grid_search.best_estimator_
best_lgb_pred = best_lgb.predict(X_test)
best_lgb_acc = accuracy_score(y_test, best_lgb_pred)
best_lgb_f1 = f1_score(y_test, best_lgb_pred, average='macro')

print(f"  调优后测试集 Accuracy: {best_lgb_acc:.4f}")
print(f"  调优后测试集 Macro-F1: {best_lgb_f1:.4f}")

# 保存调优后模型
with open(os.path.join(MODEL_DIR, 'lightgbm_tuned.pkl'), 'wb') as f:
    pickle.dump(best_lgb, f)
print(f"  ✓ 调优后模型已保存: lightgbm_tuned.pkl")

# 保存调优结果
tuning_results = pd.DataFrame(grid_search.cv_results_)
tuning_results.to_csv(os.path.join(OUTPUT_DIR, 'lgb_tuning_results.csv'), index=False)

# ============================================================
# 9. 高危岗位识别结果
# ============================================================
print("\n【9】高危岗位识别结果...")

# 用最佳模型对全量数据预测
X_full = pd.read_csv(os.path.join(DATA_DIR, 'X_full.csv')).fillna(0)
y_full_reg = pd.read_csv(os.path.join(DATA_DIR, 'y_regression.csv')).squeeze()

full_pred = best_lgb.predict(X_full)
full_pred_proba = best_lgb.predict_proba(X_full)

# 加载原始数据获取Job Title
raw_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'ai_job_trends_dataset.csv'))

# 创建结果DataFrame
result_df = raw_df.copy()
result_df['predicted_risk'] = full_pred
result_df['pred_high_risk_prob'] = full_pred_proba[:, 2]
result_df['actual_risk_category'] = pd.cut(result_df['Automation Risk (%)'],
                                            bins=[-0.01, 30, 60, 100],
                                            labels=[0, 1, 2])

# 识别高危岗位 (预测为High Risk且概率>0.5)
high_risk_jobs = result_df[result_df['predicted_risk'] == 2].copy()
high_risk_jobs_sorted = high_risk_jobs.sort_values('pred_high_risk_prob', ascending=False)

print(f"\n  预测高危岗位数量: {len(high_risk_jobs)} ({len(high_risk_jobs)/len(result_df)*100:.1f}%)")
print(f"\n--- 高危岗位 Top 20 (按预测概率排序) ---")
top20 = high_risk_jobs_sorted.head(20)
for _, row in top20.iterrows():
    print(f"  {row['Job Title']:45s}  行业={row['Industry']:15s}  实际风险={row['Automation Risk (%)']:6.1f}%  预测高危概率={row['pred_high_risk_prob']:.4f}")

# 保存高危岗位清单
high_risk_jobs_sorted.to_csv(os.path.join(OUTPUT_DIR, 'high_risk_jobs_list.csv'), index=False)
print(f"\n  ✓ 高危岗位清单已保存: high_risk_jobs_list.csv")

# 图18: 高危岗位行业分布
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 18a: 高危岗位行业分布
industry_risk = high_risk_jobs['Industry'].value_counts()
axes[0].barh(industry_risk.index, industry_risk.values, color='#e74c3c', alpha=0.8, edgecolor='white')
for i, (idx, val) in enumerate(industry_risk.items()):
    axes[0].text(val + 10, i, str(val), va='center', fontsize=10)
axes[0].set_xlabel('Count', fontsize=12)
axes[0].set_title('高危岗位行业分布', fontsize=14)

# 18b: 高危岗位AI Impact Level分布
impact_risk = high_risk_jobs['AI Impact Level'].value_counts()
colors_impact = ['#2ecc71', '#f39c12', '#e74c3c']
axes[1].pie(impact_risk.values, labels=impact_risk.index, autopct='%1.1f%%',
            colors=colors_impact, startangle=90, textprops={'fontsize': 11})
axes[1].set_title('高危岗位AI冲击等级分布', fontsize=14)

plt.suptitle('图18: 高危岗位特征分布', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig18_high_risk_distribution.png'))
plt.close()
print(">>> 图18已保存: fig18_high_risk_distribution.png")

# ============================================================
# 10. 交叉验证
# ============================================================
print("\n【10】5折交叉验证...")

cv_scores = cross_val_score(best_lgb, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=1)
print(f"  5-Fold CV Macro-F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  各折得分: {cv_scores.round(4)}")

# ============================================================
# 11. 总结
# ============================================================
print("\n" + "=" * 70)
print("  分类建模完成!")
print("=" * 70)

print(f"""
  【分类建模总结】

  1. 模型对比:
     - LightGBM:  Accuracy={lgb_acc:.4f}, Macro-F1={lgb_f1:.4f}
     - XGBoost:   Accuracy={xgb_acc:.4f}, Macro-F1={xgb_f1:.4f}
     - RF:        Accuracy={rf_acc:.4f}, Macro-F1={rf_f1:.4f}

  2. 最佳模型: {best_idx}
     - 调优后测试集 Accuracy: {best_lgb_acc:.4f}
     - 调优后测试集 Macro-F1: {best_lgb_f1:.4f}
     - 5-Fold CV Macro-F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

  3. 高危岗位识别:
     - 预测高危岗位数量: {len(high_risk_jobs)} ({len(high_risk_jobs)/len(result_df)*100:.1f}%)

  4. 生成文件:
     - models/lightgbm_model.txt
     - models/xgboost_model.json
     - models/rf_model.pkl
     - models/lightgbm_tuned.pkl
     - outputs/model_comparison.csv
     - outputs/high_risk_jobs_list.csv
     - outputs/lgb_feature_importance.csv
     - outputs/fig14-18 共5张图
""")
