"""
3地域 × 特徴量ペア散布図 (K-Means クラスタ色塗り)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rcParams["font.family"] = "Hiragino Sans"
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.expanduser("~/Documents/rice_research/")
FEATURES = ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"]
LABEL_COL = "mode"
RICE_CLASS = 3
K = 5
RANDOM_STATE = 42
SAMPLE_N = 3000  # 散布図用サンプル数（多すぎると重いので間引き）

REGIONS = {
    "つくばみらい市": "tsukubamirai_features.csv",
    "稲敷市":         "inashiki_features.csv",
    "笠間市":         "kasama_features.csv",
}

PAIR = ("VH_min", "NDVI_grow")  # x軸, y軸

CLUSTER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def run_kmeans(df):
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    y_true = (df[LABEL_COL].values == RICE_CLASS).astype(int)
    rice_ratio = np.array(
        [y_true[labels == c].mean() if (labels == c).any() else 0.0 for c in range(K)]
    )
    rice_cluster = int(np.argmax(rice_ratio))
    return labels, rice_cluster, rice_ratio


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"K-Means (k={K}) クラスタリング結果  [{PAIR[0]} vs {PAIR[1]}]",
        fontsize=15, y=1.02
    )

    for ax, (region, fname) in zip(axes, REGIONS.items()):
        fpath = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(fpath).dropna(subset=FEATURES + [LABEL_COL])

        labels, rice_cluster, rice_ratio = run_kmeans(df)

        # サンプリング
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(df), size=min(SAMPLE_N, len(df)), replace=False)
        df_s = df.iloc[idx].reset_index(drop=True)
        labels_s = labels[idx]

        x = df_s[PAIR[0]].values
        y = df_s[PAIR[1]].values

        for c in range(K):
            mask = labels_s == c
            marker = "*" if c == rice_cluster else "o"
            size   = 60  if c == rice_cluster else 20
            alpha  = 0.85 if c == rice_cluster else 0.45
            ax.scatter(
                x[mask], y[mask],
                color=CLUSTER_COLORS[c],
                marker=marker, s=size, alpha=alpha,
                label=f"C{c} (水田比率 {rice_ratio[c]:.2f})" + (" ★水田" if c == rice_cluster else ""),
            )

        ax.set_title(region, fontsize=13)
        ax.set_xlabel(PAIR[0], fontsize=11)
        ax.set_ylabel(PAIR[1], fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.expanduser("~/Downloads/scatter_kmeans_3regions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"保存完了: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
