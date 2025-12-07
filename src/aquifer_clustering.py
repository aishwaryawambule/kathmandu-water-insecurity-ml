"""
K-Means clustering of aquifer wells based on hydrogeological characteristics
and depletion patterns
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

os.makedirs('../outputs', exist_ok=True)

# Load aquifer data
df_aq = pd.read_csv("../data/raw/kathmandu_aquifer.csv")

# Features for clustering: hydrogeological + depletion indicators
features = [
    'well_depth_m', 
    'water_table_depth_m', 
    'tds_mg_per_L', 
    'nitrate_mg_per_L', 
    'transmissivity_m2_per_day', 
    'hydraulic_conductivity_m_per_day',
    'depletion_rate_m_per_year',  # KEY: depletion indicator
    'years_depleting'              # KEY: temporal dimension
]

X = df_aq[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Determine optimal k using multiple metrics
# -----------------------------
# We iterate through a range of cluster numbers (k=2 to 10) to find the optimal number of clusters.
# Metrics used:
# - Silhouette Score: Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
# - Calinski-Harabasz Score: Ratio of dispersion between and within clusters.
# - Davies-Bouldin Score: Average similarity measure of each cluster with its most similar cluster.
cluster_range = range(2, 11)
metrics = []

print("\n=== K-Means Clustering: Finding Optimal k ===")
print("k  | Silhouette | Calinski-Harabasz | Davies-Bouldin | Inertia")
print("-" * 70)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    inertia = kmeans.inertia_
    
    metrics.append({
        'k': k, 
        'silhouette': sil, 
        'ch': ch, 
        'db': db,
        'inertia': inertia
    })
    print(f"{k:2d} | {sil:10.4f} | {ch:17.2f} | {db:14.4f} | {inertia:10.2f}")

metrics_df = pd.DataFrame(metrics)

# Rank-based selection (higher is better for sil & ch, lower for db)
# We combine the ranks of the three metrics to select the best 'k'.
metrics_df['sil_rank'] = metrics_df['silhouette'].rank(ascending=False)
metrics_df['ch_rank'] = metrics_df['ch'].rank(ascending=False)
metrics_df['db_rank'] = metrics_df['db'].rank(ascending=True)
metrics_df['total_rank'] = (
    metrics_df['sil_rank'] + 
    metrics_df['ch_rank'] + 
    metrics_df['db_rank']
)

best_k = int(metrics_df.loc[metrics_df['total_rank'].idxmin(), 'k'])
print(f"\nOptimal k selected: {best_k}")

# -----------------------------
# Cluster with best k
# -----------------------------
# Apply K-Means with the selected optimal k to assign clusters to each well.
best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df_aq['aquifer_cluster'] = best_kmeans.fit_predict(X_scaled)

# -----------------------------
# Cluster statistics
# -----------------------------
# Analyze the characteristics of each cluster to understand what they represent (e.g., "High Depletion", "Stable").
print(f"\n=== Cluster Statistics (k={best_k}) ===")
cluster_summary = df_aq.groupby('aquifer_cluster').agg({
    'water_table_depth_m': ['mean', 'std'],
    'depletion_rate_m_per_year': ['mean', 'std'],
    'tds_mg_per_L': ['mean', 'std'],
    'aquifer_id': 'count'
}).round(2)
cluster_summary.columns = ['_'.join(col) for col in cluster_summary.columns]
# -----------------------------
# Name Clusters
# -----------------------------
# We assign meaningful names to clusters based on their average characteristics (Depletion Rate and Water Table Depth).
# This makes the results easier to interpret.
def name_cluster(row):
    # High depletion rate (> 1.5 m/year)
    if row['depletion_rate_m_per_year_mean'] > 1.5:
        return "Critical Depletion Zone"
    # Deep water table (> 20m) but moderate depletion
    elif row['water_table_depth_m_mean'] > 20:
        return "Deep Aquifer / High Stress"
    # Shallow water table (< 10m) and low depletion
    elif row['water_table_depth_m_mean'] < 10 and row['depletion_rate_m_per_year_mean'] < 1.0:
        return "Shallow / Stable"
    else:
        return "Moderate Stress"

cluster_summary['cluster_name'] = cluster_summary.apply(name_cluster, axis=1)
print("\n=== Cluster Names ===")
print(cluster_summary[['cluster_name']])

# Map names back to the main dataframe
cluster_name_map = cluster_summary['cluster_name'].to_dict()
df_aq['cluster_name'] = df_aq['aquifer_cluster'].map(cluster_name_map)

# Save clustered data
# This file will be used in the next step to link households to aquifer characteristics.
df_aq.to_csv("../data/processed/kathmandu_aquifer_clustered.csv", index=False)
print(f"\nSaved: kathmandu_aquifer_clustered.csv ({len(df_aq)} wells)")

# -----------------------------
# Clustering Validation Summary
# -----------------------------
print("\n=== Clustering Validation Metrics ===")
final_labels = df_aq['aquifer_cluster']

final_silhouette = silhouette_score(X_scaled, final_labels)
final_ch = calinski_harabasz_score(X_scaled, final_labels)
final_db = davies_bouldin_score(X_scaled, final_labels)

print(f"Silhouette Score       : {final_silhouette:.4f}  (closer to 1 is better)")
print(f"Calinski-Harabasz Score: {final_ch:.2f}  (higher is better)")
print(f"Davies-Bouldin Score   : {final_db:.4f}  (lower is better)")
print(f"Inertia (WCSS)         : {best_kmeans.inertia_:.2f}  (lower is better)")

cluster_sizes = df_aq['aquifer_cluster'].value_counts().sort_index()
print("\nCluster Size Distribution:")
print(cluster_sizes)

# Save validation summary
validation_summary = pd.DataFrame({
    'Metric': ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Inertia'],
    'Value': [final_silhouette, final_ch, final_db, best_kmeans.inertia_]
})
validation_summary.to_csv('../outputs/cluster_validation_summary.csv', index=False)
print("Saved: outputs/cluster_validation_summary.csv")

# -----------------------------
# Validation Metrics Plot for different k
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(metrics_df['k'], metrics_df['silhouette'], marker='o', label='Silhouette (higher better)')
plt.plot(metrics_df['k'], metrics_df['ch'], marker='s', label='Calinski-Harabasz (higher better)')
plt.plot(metrics_df['k'], metrics_df['db'], marker='^', label='Davies-Bouldin (lower better)')
plt.axvline(best_k, color='red', linestyle='--', label=f'Selected k={best_k}')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Metric Value')
plt.title('Clustering Validation Metrics vs. Number of Clusters')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/cluster_validation_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: outputs/cluster_validation_metrics.png")

# -----------------------------
# PCA Projection Plot
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=df_aq['aquifer_cluster'], 
                     cmap='Set2', 
                     s=50, 
                     alpha=0.7,
                     edgecolors='black',
                     linewidth=0.5)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title(f'Aquifer Clusters (k={best_k}) - PCA Projection')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/aquifer_clusters_pca.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: outputs/aquifer_clusters_pca.png")

# -----------------------------
# Cluster characteristics heatmap
# -----------------------------
cluster_chars = df_aq.groupby('aquifer_cluster')[features].mean()
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_chars.T, annot=True, fmt='.2f', cmap='RdYlGn_r', 
            cbar_kws={'label': 'Mean Value'})
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.title('Cluster Characteristics (Mean Values)')
plt.tight_layout()
plt.savefig('../outputs/cluster_characteristics.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: outputs/cluster_characteristics.png")

print("\n=== Clustering Complete ===")
