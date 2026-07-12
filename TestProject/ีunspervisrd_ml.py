"""
unsupervised_ml.py
==================
Unsupervised ML for Power Plant Predictive Maintenance
Algorithms:
  - Apriori  (Association Rules)
  - K-Means, DBSCAN, Agglomerative  (Clustering)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.cluster       import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold      import TSNE
from sklearn.metrics       import (silhouette_score, davies_bouldin_score,
                                   calinski_harabasz_score)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing     import TransactionEncoder

plt.rcParams.update({
    "axes.facecolor"   : "#F8F9FA",
    "figure.facecolor" : "white",
    "axes.grid"        : True,
    "grid.color"       : "#E0E0E0",
    "grid.linewidth"   : 0.6,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "DejaVu Sans",
})

PALETTE    = [plt.cm.tab10(i) for i in range(10)]
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

# PCA
pca   = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"  [PCA] {X_scaled.shape[1]} features → {X_pca.shape[1]} components "
      f"({pca.explained_variance_ratio_.sum():.2%} variance)")

# t-SNE
print("  [t-SNE] fitting ...")
tsne   = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=500)
X_tsne = tsne.fit_transform(X_pca[:3000])

# ==============================================================================
# 2. APRIORI
# ==============================================================================

print("\n  [Apriori] discretizing ...")

APRIORI_COLS = ["temperature_C", "vibration_mms", "oil_temp_C",
                "pressure_bar", "efficiency_pct"]
N_BINS     = 3
BIN_LABELS = ["Low", "Mid", "High"]

kbd    = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile")
binned = kbd.fit_transform(df[APRIORI_COLS]).astype(int)

transactions = []
for row in binned:
    items = [f"{APRIORI_COLS[j]}_{BIN_LABELS[row[j]]}"
             for j in range(len(APRIORI_COLS))]
    transactions.append(items)

te    = TransactionEncoder()
te_df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

freq_items = apriori(te_df, min_support=0.10, use_colnames=True)
rules      = association_rules(freq_items, metric="lift",
                                min_threshold=1.2, num_itemsets=len(freq_items))
rules      = rules.sort_values("lift", ascending=False).reset_index(drop=True)

print(f"  [Apriori] frequent itemsets: {len(freq_items)}")
print(f"  [Apriori] rules (lift≥1.2) : {len(rules)}")

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
dbscan_labels = {}
agglo_labels  = {}

# K-Means
for k in [2, 3, 4]:
    model  = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_pca)
    kmeans_labels[k] = labels
    m = cluster_metrics(X_pca, labels)
    m.update({"Model": f"K-Means (k={k})", "n_clusters": k, "noise_pct": 0.0})
    cluster_rows.append(m)
    print(f"  [K-Means k={k}]  Silhouette={m['Silhouette']:.4f}")

# DBSCAN
for eps, min_s in [(0.5, 5), (1.0, 5), (1.5, 10)]:
    model  = DBSCAN(eps=eps, min_samples=min_s)
    labels = model.fit_predict(X_pca)
    dbscan_labels[f"eps={eps}"] = labels
    n_cls  = len(set(labels)) - (1 if -1 in labels else 0)
    noise  = (labels == -1).sum() / len(labels) * 100
    m = cluster_metrics(X_pca, labels)
    m.update({"Model": f"DBSCAN (eps={eps})", "n_clusters": n_cls,
              "noise_pct": round(noise, 2)})
    cluster_rows.append(m)
    print(f"  [DBSCAN eps={eps}]  clusters={n_cls}  noise={noise:.1f}%")

# Agglomerative
for k, lm in [(2, "ward"), (3, "ward"), (3, "complete")]:
    model  = AgglomerativeClustering(n_clusters=k, linkage=lm)
    labels = model.fit_predict(X_pca)
    agglo_labels[f"k={k}_{lm}"] = labels
    m = cluster_metrics(X_pca, labels)
    m.update({"Model": f"Agglomerative (k={k},{lm})", "n_clusters": k,
              "noise_pct": 0.0})
    cluster_rows.append(m)
    print(f"  [Agglomerative k={k} {lm}]  Silhouette={m['Silhouette']:.4f}")

cluster_results = (
    pd.DataFrame(cluster_rows)
    [["Model","n_clusters","noise_pct","Silhouette","Davies-Bouldin","Calinski-Harabasz"]]
    .sort_values("Silhouette", ascending=False).reset_index(drop=True)
)

# ==============================================================================
# 4. PRINT TABLES
# ==============================================================================

print("\n\n" + "="*70)
print("  CLUSTERING RESULTS  (sorted by Silhouette ↓)")
print("="*70)
print(cluster_results.to_string(index=True))

print("\n\n" + "="*70)
print("  TOP 10 ASSOCIATION RULES  (sorted by lift ↓)")
print("="*70)
print(rules[["antecedents","consequents","support","confidence","lift"]]
      .head(10).to_string(index=True))

# ==============================================================================
# 5. EXPORT CSV  (2 files)
# ==============================================================================

cluster_results.to_csv(f"{OUTPUT_DIR}/unsup_cluster_results.csv", index=False)
print(f"\n  [Saved] {OUTPUT_DIR}/unsup_cluster_results.csv")

rules_out = rules[["antecedents","consequents","support","confidence",
                   "lift","leverage","conviction"]].copy()
rules_out["antecedents"] = rules_out["antecedents"].apply(lambda x: ", ".join(list(x)))
rules_out["consequents"] = rules_out["consequents"].apply(lambda x: ", ".join(list(x)))
rules_out.to_csv(f"{OUTPUT_DIR}/unsup_apriori_rules.csv", index=False)
print(f"  [Saved] {OUTPUT_DIR}/unsup_apriori_rules.csv")

# ==============================================================================
# 6. FIGURE 1 — Clustering แบ่งตาม Algorithm
# ==============================================================================

def section_title(fig, text, y_pos):
    fig.text(0.5, y_pos, text, ha="center", fontsize=13, fontweight="bold",
             color="white",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#2C3E50", alpha=0.9))

fig1 = plt.figure(figsize=(22, 26))
fig1.suptitle("Unsupervised ML — Clustering",
              fontsize=18, fontweight="bold", y=0.995)
gs1  = gridspec.GridSpec(4, 3, figure=fig1, hspace=0.55, wspace=0.35)

section_title(fig1, "── K-Means Clustering ──",                      0.965)
section_title(fig1, "── DBSCAN Clustering ──",                        0.720)
section_title(fig1, "── Agglomerative (Hierarchical) Clustering ──", 0.475)

# ROW 0 — K-Means
ax = fig1.add_subplot(gs1[0, 0])
for k_val in range(2):
    idx = kmeans_labels[2] == k_val
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[PALETTE[k_val]],
               alpha=0.3, s=5, label=f"Cluster {k_val}", rasterized=True)
ax.set_title("K-Means (k=2) on PCA", fontsize=11, fontweight="bold")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(fontsize=8, markerscale=3)

ax = fig1.add_subplot(gs1[0, 1])
for k_val in range(3):
    idx = kmeans_labels[3] == k_val
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[PALETTE[k_val]],
               alpha=0.3, s=5, label=f"Cluster {k_val}", rasterized=True)
ax.set_title("K-Means (k=3) on PCA", fontsize=11, fontweight="bold")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(fontsize=8, markerscale=3)

ax = fig1.add_subplot(gs1[0, 2])
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10)
            .fit(X_pca).inertia_ for k in range(2, 9)]
ax.plot(range(2, 9), inertias, marker="o", color=PALETTE[0], linewidth=2)
ax.set_title("K-Means Elbow Curve", fontsize=11, fontweight="bold")
ax.set_xlabel("Number of Clusters (k)"); ax.set_ylabel("Inertia")

# ROW 1 — DBSCAN
for col_idx, eps_key in enumerate(["eps=0.5", "eps=1.0", "eps=1.5"]):
    ax  = fig1.add_subplot(gs1[1, col_idx])
    lbs = dbscan_labels[eps_key]
    for lb in sorted(set(lbs)):
        idx   = lbs == lb
        color = "#AAAAAA" if lb == -1 else PALETTE[lb % 10]
        name  = "Noise"  if lb == -1 else f"Cluster {lb}"
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[color],
                   alpha=0.3, s=5, label=name, rasterized=True)
    n_cls   = len(set(lbs)) - (1 if -1 in lbs else 0)
    noise   = (lbs == -1).sum() / len(lbs) * 100
    eps_val = eps_key.split("=")[1]
    ax.set_title(f"DBSCAN (eps={eps_val})  clusters={n_cls}  noise={noise:.1f}%",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=7, markerscale=3)

# ROW 2 — Agglomerative
for col_idx, (key, title) in enumerate([
    ("k=2_ward",     "Agglomerative (k=2, ward)"),
    ("k=3_ward",     "Agglomerative (k=3, ward)"),
    ("k=3_complete", "Agglomerative (k=3, complete)"),
]):
    ax  = fig1.add_subplot(gs1[2, col_idx])
    lbs = agglo_labels[key]
    for k_val in range(len(set(lbs))):
        idx = lbs == k_val
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[PALETTE[k_val]],
                   alpha=0.3, s=5, label=f"Cluster {k_val}", rasterized=True)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=8, markerscale=3)

# ROW 3 — Summary
ax = fig1.add_subplot(gs1[3, :2])
valid      = cluster_results.dropna(subset=["Silhouette"])
bar_colors = [PALETTE[0] if "K-Means"     in m
              else PALETTE[1] if "DBSCAN" in m
              else PALETTE[2]
              for m in valid["Model"]]
bars = ax.barh(valid["Model"], valid["Silhouette"],
               color=bar_colors, edgecolor="white", height=0.6)
for bar, val in zip(bars, valid["Silhouette"]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
ax.set_title("Silhouette Score — All Models Comparison (↑ better)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Silhouette Score")
ax.legend(handles=[Patch(color=PALETTE[0], label="K-Means"),
                   Patch(color=PALETTE[1], label="DBSCAN"),
                   Patch(color=PALETTE[2], label="Agglomerative")],
          fontsize=9, loc="lower right")

ax = fig1.add_subplot(gs1[3, 2])
sample_idx = np.random.choice(len(X_pca), 300, replace=False)
Z = linkage(X_pca[sample_idx], method="ward")
dendrogram(Z, ax=ax, no_labels=True,
           color_threshold=0.7 * max(Z[:, 2]),
           above_threshold_color="#AAAAAA")
ax.set_title("Dendrogram (Ward, n=300)", fontsize=11, fontweight="bold")
ax.set_xlabel("Samples"); ax.set_ylabel("Distance")

fig1.savefig(f"{OUTPUT_DIR}/unsup_clustering.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  [Saved] {OUTPUT_DIR}/unsup_clustering.png")

# ==============================================================================
# 7. FIGURE 2 — Apriori Association Rules
# ==============================================================================

top_rules = rules.head(20)

fig2 = plt.figure(figsize=(20, 14))
fig2.suptitle("Unsupervised ML — Apriori Association Rules",
              fontsize=18, fontweight="bold", y=0.98)
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.35)

# 2a. Top rules by Lift
ax = fig2.add_subplot(gs2[0, :2])
labels_rules = [
    f"{', '.join(list(r.antecedents))} → {', '.join(list(r.consequents))}"
    for _, r in top_rules.iterrows()
]
ax.barh(range(len(top_rules)), top_rules["lift"].values,
        color=[PALETTE[i % 10] for i in range(len(top_rules))],
        edgecolor="white", height=0.6)
ax.set_yticks(range(len(top_rules)))
ax.set_yticklabels(labels_rules, fontsize=7)
ax.set_title("Top 20 Rules by Lift (↑ stronger association)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Lift")

# 2b. Support vs Confidence scatter
ax = fig2.add_subplot(gs2[0, 2])
sc = ax.scatter(rules["support"], rules["confidence"],
                c=rules["lift"], cmap="YlOrRd",
                alpha=0.7, s=40, edgecolors="white", linewidths=0.5)
plt.colorbar(sc, ax=ax, label="Lift")
ax.set_title("Support vs Confidence\n(colored by Lift)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Support"); ax.set_ylabel("Confidence")

# 2c. Support distribution
ax = fig2.add_subplot(gs2[1, 0])
ax.hist(rules["support"], bins=30, color=PALETTE[0],
        edgecolor="white", alpha=0.8)
ax.set_title("Support Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Support"); ax.set_ylabel("Count")

# 2d. Confidence distribution
ax = fig2.add_subplot(gs2[1, 1])
ax.hist(rules["confidence"], bins=30, color=PALETTE[1],
        edgecolor="white", alpha=0.8)
ax.set_title("Confidence Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Confidence"); ax.set_ylabel("Count")

# 2e. Lift distribution
ax = fig2.add_subplot(gs2[1, 2])
ax.hist(rules["lift"], bins=30, color=PALETTE[2],
        edgecolor="white", alpha=0.8)
ax.axvline(1.0, color="#E8564C", linestyle="--",
           linewidth=2, label="Lift=1 (no association)")
ax.set_title("Lift Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Lift"); ax.set_ylabel("Count")
ax.legend(fontsize=9)

fig2.savefig(f"{OUTPUT_DIR}/unsup_apriori.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  [Saved] {OUTPUT_DIR}/unsup_apriori.png")

print("\n  Done.")