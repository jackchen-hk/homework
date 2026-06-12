# -*- coding: utf-8 -*-
"""
=================================================================
  基于多维度特征的劳动力AI替代风险挖掘与高危岗位识别
  Step 4: 聚类分析 (成员C负责)
  方法: K-Means, GMM, 层次聚类, DBSCAN
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
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

print("=" * 70)
print("  第四步：聚类分析")
print("=" * 70)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n【1】加载数据...")

X_full = pd.read_csv(os.path.join(DATA_DIR, 'X_full.csv')).fillna(0)
y_full_reg = pd.read_csv(os.path.join(DATA_DIR, 'y_regression.csv')).squeeze()
raw_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'ai_job_trends_dataset.csv'))

with open(os.path.join(DATA_DIR, 'feature_columns.pkl'), 'rb') as f:
    feature_cols = pickle.load(f)

print(f"  数据形状: {X_full.shape}")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)
print(f"  标准化完成")

# ============================================================
# 2. 降维可视化
# ============================================================
print("\n【2】降维可视化 (PCA + t-SNE)...")

# PCA降维
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print(f"  PCA解释方差比: {pca.explained_variance_ratio_.round(4)}")
print(f"  PCA总解释方差: {pca.explained_variance_ratio_.sum():.4f}")

# t-SNE (抽样计算, 全量太慢)
sample_size = 5000
sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_scaled), sample_size, replace=False)
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled[sample_idx])
print(f"  t-SNE降维完成 (抽样{sample_size}条)")

# 风险类别标签
risk_category = pd.cut(y_full_reg, bins=[-0.01, 30, 60, 100], labels=[0, 1, 2])

# 图19: PCA散点图 (按风险类别着色)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

risk_colors = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
risk_labels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

for cat in [0, 1, 2]:
    mask = risk_category == cat
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=risk_colors[cat],
                    label=risk_labels[cat], alpha=0.3, s=8)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
axes[0].set_title('PCA降维 (按风险类别着色)', fontsize=14)
axes[0].legend(fontsize=10)

# t-SNE散点图
risk_category_sample = risk_category.iloc[sample_idx]
for cat in [0, 1, 2]:
    mask = risk_category_sample == cat
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=risk_colors[cat],
                    label=risk_labels[cat], alpha=0.3, s=8)
axes[1].set_xlabel('t-SNE 1', fontsize=12)
axes[1].set_ylabel('t-SNE 2', fontsize=12)
axes[1].set_title(f't-SNE降维 (抽样{sample_size}条)', fontsize=14)
axes[1].legend(fontsize=10)

plt.suptitle('图19: 降维可视化', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig19_dim_reduction.png'))
plt.close()
print(">>> 图19已保存: fig19_dim_reduction.png")

# ============================================================
# 3. K-Means 聚类 (肘部法 + 轮廓系数)
# ============================================================
print("\n【3】K-Means 最优K值选择...")

k_range = range(2, 11)
inertias = []
silhouettes = []
ch_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels, sample_size=5000, random_state=RANDOM_STATE))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))
    print(f"  K={k}: Inertia={km.inertia_:.0f}, Silhouette={silhouettes[-1]:.4f}, CH={ch_scores[-1]:.1f}")

# 图20: 肘部法 + 轮廓系数
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia', fontsize=12)
axes[0].set_title('肘部法 (Elbow Method)', fontsize=14)
axes[0].grid(alpha=0.3)

axes[1].plot(k_range, silhouettes, 'ro-', linewidth=2, markersize=8)
best_k = list(k_range)[np.argmax(silhouettes)]
axes[1].axvline(x=best_k, color='green', linestyle='--', label=f'Best K={best_k}')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('轮廓系数法', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

axes[2].plot(k_range, ch_scores, 'go-', linewidth=2, markersize=8)
axes[2].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[2].set_ylabel('Calinski-Harabasz Score', fontsize=12)
axes[2].set_title('CH指数法', fontsize=14)
axes[2].grid(alpha=0.3)

plt.suptitle('图20: K-Means 最优K值选择', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig20_optimal_k.png'))
plt.close()
print(f">>> 图20已保存: fig20_optimal_k.png")
print(f"  建议最优K值: {best_k} (基于轮廓系数)")

# ============================================================
# 4. K-Means 最终聚类
# ============================================================
print(f"\n【4】K-Means 聚类 (K={best_k})...")

kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
kmeans_labels = kmeans_final.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, kmeans_labels, sample_size=5000, random_state=RANDOM_STATE)
ch_score = calinski_harabasz_score(X_scaled, kmeans_labels)
db_score = davies_bouldin_score(X_scaled, kmeans_labels)

print(f"  Silhouette Score: {sil_score:.4f}")
print(f"  Calinski-Harabasz Score: {ch_score:.1f}")
print(f"  Davies-Bouldin Score: {db_score:.4f}")

# 保存模型
with open(os.path.join(MODEL_DIR, 'kmeans_model.pkl'), 'wb') as f:
    pickle.dump(kmeans_final, f)

# ============================================================
# 5. GMM 聚类
# ============================================================
print(f"\n【5】GMM 聚类...")

# 选择最优n_components
bic_scores = []
aic_scores = []
for n in range(2, 8):
    gmm = GaussianMixture(n_components=n, random_state=RANDOM_STATE, covariance_type='full')
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))
    print(f"  n={n}: BIC={gmm.bic(X_scaled):.0f}, AIC={gmm.aic(X_scaled):.0f}")

best_n_gmm = list(range(2, 8))[np.argmin(bic_scores)]
print(f"  最优n_components: {best_n_gmm} (基于BIC)")

gmm_final = GaussianMixture(n_components=best_n_gmm, random_state=RANDOM_STATE, covariance_type='full')
gmm_labels = gmm_final.fit_predict(X_scaled)

gmm_sil = silhouette_score(X_scaled, gmm_labels, sample_size=5000, random_state=RANDOM_STATE)
print(f"  GMM Silhouette Score: {gmm_sil:.4f}")

# 保存模型
with open(os.path.join(MODEL_DIR, 'gmm_model.pkl'), 'wb') as f:
    pickle.dump(gmm_final, f)

# ============================================================
# 6. 聚类结果分析
# ============================================================
print(f"\n【6】聚类结果分析 (K-Means)...")

result_df = raw_df.copy()
result_df['cluster'] = kmeans_labels
result_df['Automation Risk (%)'] = y_full_reg

# 各聚类的特征均值
cluster_stats = result_df.groupby('cluster').agg({
    'Automation Risk (%)': ['mean', 'std', 'min', 'max'],
    'Median Salary (USD)': 'mean',
    'Experience Required (Years)': 'mean',
    'Remote Work Ratio (%)': 'mean',
    'Gender Diversity (%)': 'mean',
    'Job Title': 'count'
}).round(2)

print("\n--- 各聚类特征统计 ---")
print(cluster_stats.to_string())

# 各聚类的行业分布
print("\n--- 各聚类行业分布 ---")
cluster_industry = pd.crosstab(result_df['cluster'], result_df['Industry'], normalize='index').round(3) * 100
print(cluster_industry.to_string())

# 各聚类的AI Impact Level分布
print("\n--- 各聚类AI冲击等级分布 ---")
cluster_impact = pd.crosstab(result_df['cluster'], result_df['AI Impact Level'], normalize='index').round(3) * 100
print(cluster_impact.to_string())

# 识别高危聚类
cluster_risk_mean = result_df.groupby('cluster')['Automation Risk (%)'].mean()
high_risk_cluster = cluster_risk_mean.idxmax()
low_risk_cluster = cluster_risk_mean.idxmin()
print(f"\n  高危聚类: Cluster {high_risk_cluster} (平均风险={cluster_risk_mean[high_risk_cluster]:.2f}%)")
print(f"  低危聚类: Cluster {low_risk_cluster} (平均风险={cluster_risk_mean[low_risk_cluster]:.2f}%)")

# 高危聚类中的Top岗位
high_cluster_jobs = result_df[result_df['cluster'] == high_risk_cluster]
print(f"\n--- 高危聚类(Cluster {high_risk_cluster}) Top 15 岗位 ---")
print(high_cluster_jobs['Job Title'].value_counts().head(15).to_string())

# 图21: 聚类结果可视化
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 21a: PCA + 聚类标签
cluster_colors = plt.cm.Set1(np.linspace(0, 1, best_k))
for c in range(best_k):
    mask = kmeans_labels == c
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=[cluster_colors[c]],
                    label=f'Cluster {c} (n={mask.sum()})', alpha=0.3, s=8)
axes[0].set_xlabel('PC1', fontsize=12)
axes[0].set_ylabel('PC2', fontsize=12)
axes[0].set_title(f'K-Means 聚类结果 (K={best_k})', fontsize=14)
axes[0].legend(fontsize=9, markerscale=5)

# 21b: 各聚类的Automation Risk箱线图
cluster_risk_data = [result_df[result_df['cluster'] == c]['Automation Risk (%)'].values for c in range(best_k)]
bp = axes[1].boxplot(cluster_risk_data, patch_artist=True, labels=[f'C{c}' for c in range(best_k)])
for patch, color in zip(bp['boxes'], cluster_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_xlabel('Cluster', fontsize=12)
axes[1].set_ylabel('Automation Risk (%)', fontsize=12)
axes[1].set_title('各聚类自动化风险分布', fontsize=14)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('图21: K-Means聚类结果', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig21_kmeans_clusters.png'))
plt.close()
print(">>> 图21已保存: fig21_kmeans_clusters.png")

# 图22: 聚类画像雷达图
print("\n【6.1】聚类画像雷达图...")

# 选取关键维度
radar_features = ['Median Salary (USD)', 'Experience Required (Years)',
                   'Remote Work Ratio (%)', 'Gender Diversity (%)', 'Automation Risk (%)']

# 标准化到0-1范围用于雷达图
from sklearn.preprocessing import MinMaxScaler
radar_data = result_df.groupby('cluster')[radar_features].mean()
mms = MinMaxScaler()
radar_data_norm = pd.DataFrame(mms.fit_transform(radar_data), columns=radar_features, index=radar_data.index)

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
angles += angles[:1]

for c in range(best_k):
    values = radar_data_norm.loc[c].values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {c}', color=cluster_colors[c])
    ax.fill(angles, values, alpha=0.1, color=cluster_colors[c])

ax.set_xticks(angles[:-1])
ax.set_xticklabels([f.replace(' (USD)', '\n(USD)').replace(' (%)', '\n(%)') for f in radar_features], fontsize=10)
ax.set_title('图22: 聚类画像雷达图', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig22_cluster_radar.png'))
plt.close()
print(">>> 图22已保存: fig22_cluster_radar.png")

# ============================================================
# 7. 保存聚类结果
# ============================================================
print("\n【7】保存聚类结果...")

result_df.to_csv(os.path.join(OUTPUT_DIR, 'clustering_results.csv'), index=False)
cluster_stats.to_csv(os.path.join(OUTPUT_DIR, 'cluster_statistics.csv'))
print(f"  ✓ 聚类结果已保存: clustering_results.csv")
print(f"  ✓ 聚类统计已保存: cluster_statistics.csv")

# ============================================================
# 8. 总结
# ============================================================
print("\n" + "=" * 70)
print("  聚类分析完成!")
print("=" * 70)

print(f"""
  【聚类分析总结】

  1. 最优K值: {best_k} (基于轮廓系数)
     - K-Means Silhouette Score: {sil_score:.4f}
     - K-Means CH Score: {ch_score:.1f}

  2. GMM:
     - 最优n_components: {best_n_gmm}
     - GMM Silhouette Score: {gmm_sil:.4f}

  3. 高危聚类识别:
     - Cluster {high_risk_cluster} 为最高危聚类 (平均风险={cluster_risk_mean[high_risk_cluster]:.2f}%)
     - Cluster {low_risk_cluster} 为最低危聚类 (平均风险={cluster_risk_mean[low_risk_cluster]:.2f}%)

  4. 生成文件:
     - models/kmeans_model.pkl
     - models/gmm_model.pkl
     - outputs/clustering_results.csv
     - outputs/cluster_statistics.csv
     - outputs/fig19-22 共4张图
""")
