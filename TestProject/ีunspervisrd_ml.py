"""
unsupervised_ml.py
==================
Unsupervised ML for Power Plant Predictive Maintenance
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing   import StandardScaler
from sklearn.cluster         import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble        import IsolationForest
from sklearn.neighbors       import LocalOutlierFactor
from sklearn.decomposition   import PCA
from sklearn.manifold        import TSNE
from sklearn.metrics         import (silhouette_score, davies_bouldin_score,
                                     calinski_harabasz_score)

plt.rcParams.update({
    "axes.facecolor"  : "#F8F9FA",
    "figure.facecolor": "white",
    "axes.grid"       : True,
    "grid.color"      : "#E0E0E0",
    "grid.linewidth"  : 0.6,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "font.family"     : "DejaVu Sans",
})

OUTPUT_DIR = "output"

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

train_df = pd.read_csv("DataSet/unsupervised_dataset.csv")
test_df  = pd.read_csv("DataSet/unsupervised_dataset_test.csv")
df       = pd.concat([train_df, test_df], ignore_index=True)

FEATURE_COLS = [
    "operating_hours", "load_pct", "days_since_maintenance", "maintenance_count",
    "temperature_C", "pressure_bar", "vibration_mms", "rotation_speed_rpm",
    "voltage_V", "current_A", "oil_temp_C", "power_output_MW", "efficiency_pct",
]

X        = df[FEATURE_COLS]
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# 2. PCA
# ==============================================================================

pca   = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"  [PCA] {X_scaled.shape[1]} features → {X_pca.shape[1]} components "
      f"({pca.explained_variance_ratio_.sum():.2%} variance)")

pca_loadings = pd.DataFrame(
    pca.components_.T,
    index=FEATURE_COLS,
    columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
).round(4)

# t-SNE (2D for visualization)
print("  [t-SNE] fitting ...")
tsne   = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=500)
X_tsne = tsne.fit_transform(X_pca[:3000])    # sample for speed
df_tsne = df.iloc[:3000].copy()

# ==============================================================================
# 3. CLUSTERING
# ==============================================================================

def cluster_metrics(X, labels):
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return {"Silhouette": None, "Davies-Bouldin": None, "Calinski-Harabasz": None}
    return {
        "Silhouette"        : silhouette_score(X[mask], labels[mask]),
        "Davies-Bouldin"    : davies_bouldin_score(X[mask], labels[mask]),
        "Calinski-Harabasz" : calinski_harabasz_score(X[mask], labels[mask]),
    }

cluster_rows  = []
kmeans_labels = {}

for k in [2, 3, 4]:
    model  = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_pca)
    kmeans_labels[k] = labels
    m = cluster_metrics(X_pca, labels)
    m.update({"Model": f"K-Means (k={k})", "n_clusters": k, "noise_pct": 0.0})
    cluster_rows.append(m)
    print(f"  [K-Means k={k}]  Silhouette={m['Silhouette']:.4f}")

dbscan_labels = {}
for eps, min_samples in [(0.5, 5), (1.0, 5), (1.5, 10)]:
    model  = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_pca)
    dbscan_labels[f"eps={eps}"] = labels
    n_cls  = len(set(labels)) - (1 if -1 in labels else 0)
    noise  = (labels == -1).sum() / len(labels) * 100
    m = cluster_metrics(X_pca, labels)
    m.update({"Model": f"DBSCAN (eps={eps})", "n_clusters": n_cls, "noise_pct": round(noise, 2)})
    cluster_rows.append(m)
    print(f"  [DBSCAN eps={eps}]  clusters={n_cls}  noise={noise:.1f}%")

agglo_labels = {}
for k, linkage_method in [(2, "ward"), (3, "ward"), (3, "complete")]:
    model  = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = model.fit_predict(X_pca)
    agglo_labels[f"k={k}_{linkage_method}"] = labels
    m = cluster_metrics(X_pca, labels)
    m.update({"Model": f"Agglomerative (k={k}, {linkage_method})",
              "n_clusters": k, "noise_pct": 0.0})
    cluster_rows.append(m)
    print(f"  [Agglomerative k={k} {linkage_method}]  Silhouette={m['Silhouette']:.4f}")

cluster_results = (
    pd.DataFrame(cluster_rows)
    [["Model","n_clusters","noise_pct","Silhouette","Davies-Bouldin","Calinski-Harabasz"]]
    .sort_values("Silhouette", ascending=False).reset_index(drop=True)
)

# ==============================================================================
# 4. ANOMALY DETECTION
# ==============================================================================

anomaly_rows = []
iso_labels   = {}
lof_labels   = {}

for contamination in [0.05, 0.10]:
    model  = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    labels = model.fit_predict(X_scaled)
    scores = -model.score_samples(X_scaled)
    iso_labels[contamination] = labels
    n_anom = (labels == -1).sum()
    anomaly_rows.append({
        "Model"             : f"Isolation Forest (cont={contamination})",
        "Contamination"     : contamination,
        "n_anomalies"       : n_anom,
        "anomaly_pct"       : round(n_anom / len(labels) * 100, 2),
        "mean_anomaly_score": scores[labels == -1].mean().round(4),
        "mean_normal_score" : scores[labels ==  1].mean().round(4),
    })
    print(f"  [IsoForest cont={contamination}]  anomalies={n_anom}")

for n_neighbors, contamination in [(20, 0.05), (20, 0.10)]:
    model  = LocalOutlierFactor(n_neighbors=n_neighbors,
                                contamination=contamination, n_jobs=-1)
    labels = model.fit_predict(X_scaled)
    scores = -model.negative_outlier_factor_
    lof_labels[contamination] = labels
    n_anom = (labels == -1).sum()
    anomaly_rows.append({
        "Model"             : f"LOF (k={n_neighbors}, cont={contamination})",
        "Contamination"     : contamination,
        "n_anomalies"       : n_anom,
        "anomaly_pct"       : round(n_anom / len(labels) * 100, 2),
        "mean_anomaly_score": scores[labels == -1].mean().round(4),
        "mean_normal_score" : scores[labels ==  1].mean().round(4),
    })
    print(f"  [LOF cont={contamination}]  anomalies={n_anom}")

anomaly_results = pd.DataFrame(anomaly_rows).reset_index(drop=True)

# ==============================================================================
# 5. PRINT TABLES
# ==============================================================================

print("\n\n" + "="*75)
print("  CLUSTERING RESULTS  (sorted by Silhouette ↓)")
print("="*75)
print(cluster_results.to_string(index=True))

print("\n\n" + "="*75)
print("  ANOMALY DETECTION RESULTS")
print("="*75)
print(anomaly_results.to_string(index=True))

# ==============================================================================
# 6. VISUALIZATIONS
# ==============================================================================

CMAP    = plt.cm.tab10
PALETTE = [CMAP(i) for i in range(10)]

# ── Figure 1: Clustering Dashboard ──────────────────────────────────────────
fig1 = plt.figure(figsize=(20, 18))
fig1.suptitle("Unsupervised ML — Clustering", fontsize=18, fontweight="bold", y=0.98)
gs1  = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.45, wspace=0.35)

# 1a. Silhouette comparison bar
ax = fig1.add_subplot(gs1[0, :2])
valid = cluster_results.dropna(subset=["Silhouette"])
colors_bar = [PALETTE[i % len(PALETTE)] for i in range(len(valid))]
bars = ax.barh(valid["Model"], valid["Silhouette"], color=colors_bar,
               edgecolor="white", height=0.6)
for bar, val in zip(bars, valid["Silhouette"]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
ax.set_title("Silhouette Score Comparison (↑ better)", fontsize=12, fontweight="bold")
ax.set_xlabel("Silhouette Score")

# 1b. Davies-Bouldin comparison
ax = fig1.add_subplot(gs1[0, 2])
valid_db = cluster_results.dropna(subset=["Davies-Bouldin"])
ax.barh(valid_db["Model"], valid_db["Davies-Bouldin"],
        color=colors_bar[:len(valid_db)], edgecolor="white", height=0.6)
ax.set_title("Davies-Bouldin (↓ better)", fontsize=12, fontweight="bold")
ax.set_xlabel("DB Score")
ax.tick_params(axis="y", labelsize=7)

# 1c. K-Means k=2 on PCA 2D
ax = fig1.add_subplot(gs1[1, 0])
for k_val, color, label in zip(range(2), PALETTE, ["Cluster 0", "Cluster 1"]):
    idx = kmeans_labels[2] == k_val
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[color], alpha=0.3,
               s=5, label=label, rasterized=True)
ax.set_title("K-Means (k=2) on PCA", fontsize=12, fontweight="bold")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(fontsize=8, markerscale=3)

# 1d. K-Means k=3 on PCA 2D
ax = fig1.add_subplot(gs1[1, 1])
for k_val in range(3):
    idx = kmeans_labels[3] == k_val
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[PALETTE[k_val]], alpha=0.3,
               s=5, label=f"Cluster {k_val}", rasterized=True)
ax.set_title("K-Means (k=3) on PCA", fontsize=12, fontweight="bold")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(fontsize=8, markerscale=3)

# 1e. DBSCAN eps=1.0 on PCA 2D
ax = fig1.add_subplot(gs1[1, 2])
lbs = dbscan_labels["eps=1.0"]
unique_lbs = sorted(set(lbs))
for lb in unique_lbs:
    idx   = lbs == lb
    color = "#AAAAAA" if lb == -1 else PALETTE[lb % len(PALETTE)]
    name  = "Noise" if lb == -1 else f"Cluster {lb}"
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[color], alpha=0.3,
               s=5, label=name, rasterized=True)
ax.set_title("DBSCAN (eps=1.0) on PCA", fontsize=12, fontweight="bold")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(fontsize=7, markerscale=3)

# 1f. t-SNE K-Means k=3
ax = fig1.add_subplot(gs1[2, 0])
km_tsne = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_tsne)
for k_val in range(3):
    idx = km_tsne == k_val
    ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=[PALETTE[k_val]], alpha=0.4,
               s=5, label=f"Cluster {k_val}", rasterized=True)
ax.set_title("t-SNE + K-Means (k=3)", fontsize=12, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(fontsize=8, markerscale=3)

# 1g. Elbow curve (Inertia)
ax = fig1.add_subplot(gs1[2, 1])
inertias = []
k_range  = range(2, 9)
for k in k_range:
    inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca).inertia_)
ax.plot(list(k_range), inertias, marker="o", color=PALETTE[0], linewidth=2)
ax.set_title("Elbow Curve (K-Means Inertia)", fontsize=12, fontweight="bold")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")

# 1h. Dendrogram (Hierarchical)
ax = fig1.add_subplot(gs1[2, 2])
sample_idx = np.random.choice(len(X_pca), 300, replace=False)
Z = linkage(X_pca[sample_idx], method="ward")
dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.7*max(Z[:, 2]),
           above_threshold_color="#AAAAAA")
ax.set_title("Dendrogram (Ward, n=300)", fontsize=12, fontweight="bold")
ax.set_xlabel("Samples"); ax.set_ylabel("Distance")

fig1.savefig(f"{OUTPUT_DIR}/unsup_clustering.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  [Saved] {OUTPUT_DIR}/unsup_clustering.png")

# ── Figure 2: Anomaly Detection Dashboard ───────────────────────────────────
fig2 = plt.figure(figsize=(20, 14))
fig2.suptitle("Unsupervised ML — Anomaly Detection", fontsize=18, fontweight="bold", y=0.98)
gs2  = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.35)

# 2a. Anomaly count comparison
ax = fig2.add_subplot(gs2[0, 0])
ax.bar(anomaly_results["Model"], anomaly_results["n_anomalies"],
       color=[PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]], edgecolor="white")
ax.set_title("Number of Anomalies Detected", fontsize=12, fontweight="bold")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=25, labelsize=7)
for i, val in enumerate(anomaly_results["n_anomalies"]):
    ax.text(i, val + 10, str(val), ha="center", fontsize=9, fontweight="bold")

# 2b. Isolation Forest cont=0.05 on PCA
ax = fig2.add_subplot(gs2[0, 1])
lbs = iso_labels[0.05]
for lb, color, name in [(1, PALETTE[0], "Normal"), (-1, "#E8564C", "Anomaly")]:
    idx = lbs == lb
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[color], alpha=0.3 if lb==1 else 0.7,
               s=5 if lb==1 else 15, label=name, rasterized=True)
ax.set_title("Isolation Forest (cont=0.05)", fontsize=12, fontweight="bold")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(fontsize=9, markerscale=3)

# 2c. LOF cont=0.05 on PCA
ax = fig2.add_subplot(gs2[0, 2])
lbs = lof_labels[0.05]
for lb, color, name in [(1, PALETTE[0], "Normal"), (-1, "#E8564C", "Anomaly")]:
    idx = lbs == lb
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[color], alpha=0.3 if lb==1 else 0.7,
               s=5 if lb==1 else 15, label=name, rasterized=True)
ax.set_title("LOF (cont=0.05)", fontsize=12, fontweight="bold")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(fontsize=9, markerscale=3)

# 2d. Isolation Forest on t-SNE
ax = fig2.add_subplot(gs2[1, 0])
lbs_tsne = iso_labels[0.05][:3000]
for lb, color, name in [(1, PALETTE[0], "Normal"), (-1, "#E8564C", "Anomaly")]:
    idx = lbs_tsne == lb
    ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=[color], alpha=0.3 if lb==1 else 0.8,
               s=5 if lb==1 else 20, label=name, rasterized=True)
ax.set_title("t-SNE + Isolation Forest", fontsize=12, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(fontsize=9, markerscale=3)

# 2e. PCA Explained Variance
ax = fig2.add_subplot(gs2[1, 1])
var_ratio = pca.explained_variance_ratio_
cum_var   = np.cumsum(var_ratio)
x_pc = range(1, len(var_ratio) + 1)
ax.bar(x_pc, var_ratio * 100, color=PALETTE[0], alpha=0.7,
       label="Individual", edgecolor="white")
ax2_twin = ax.twinx()
ax2_twin.plot(x_pc, cum_var * 100, color="#E8564C", marker="o",
              linewidth=2, label="Cumulative")
ax2_twin.axhline(95, color="#F0A500", linestyle="--", linewidth=1.5, label="95%")
ax2_twin.set_ylabel("Cumulative Variance (%)", color="#E8564C")
ax2_twin.set_ylim(0, 105)
ax.set_title("PCA Explained Variance", fontsize=12, fontweight="bold")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Individual Variance (%)")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

# 2f. PCA Loadings Heatmap (PC1–PC3)
ax = fig2.add_subplot(gs2[1, 2])
load_plot = pca_loadings.iloc[:, :3]
im = ax.imshow(load_plot.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
ax.set_xticks(range(load_plot.shape[1]))
ax.set_xticklabels(load_plot.columns, fontsize=9)
ax.set_yticks(range(len(FEATURE_COLS)))
ax.set_yticklabels(FEATURE_COLS, fontsize=8)
for i in range(load_plot.shape[0]):
    for j in range(load_plot.shape[1]):
        ax.text(j, i, f"{load_plot.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("PCA Loadings (PC1–PC3)", fontsize=12, fontweight="bold")

fig2.savefig(f"{OUTPUT_DIR}/unsup_anomaly.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  [Saved] {OUTPUT_DIR}/unsup_anomaly.png")

# ==============================================================================
# 7. EXPORT CSV
# ==============================================================================

cluster_results.to_csv(f"{OUTPUT_DIR}/unsup_cluster_results.csv",  index=False)
anomaly_results.to_csv(f"{OUTPUT_DIR}/unsup_anomaly_results.csv",  index=False)
pca_loadings.to_csv(f"{OUTPUT_DIR}/unsup_pca_loadings.csv")

print(f"  [Saved] {OUTPUT_DIR}/unsup_cluster_results.csv")
print(f"  [Saved] {OUTPUT_DIR}/unsup_anomaly_results.csv")
print(f"  [Saved] {OUTPUT_DIR}/unsup_pca_loadings.csv")