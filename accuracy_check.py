"""
3地域 精度測定スクリプト（最新手法）
- 特徴量: VH_min, NDVI_grow, VH_winter, NDVI_flood, dNDVI, dVH（6特徴量）
- K-Means k=4
- 3地域それぞれの精度を出力
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 設定 ---
DATA_DIR = "/Users/nagatomotaito/Documents/rice_research/data/"
BASE_FEATURES = ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"]
LABEL_COL = "mode"
RICE_CLASS = 3
K = 4
RANDOM_STATE = 42

# 地域ごとのCSVファイル
REGIONS = {
    "つくばみらい市": f"{DATA_DIR}tsukubamirai_features.csv",
    "稲敷市":         f"{DATA_DIR}inashiki_features.csv",
    "笠間市":         f"{DATA_DIR}kasama_features.csv",
}


def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    union = tp + fp + fn
    return float(tp / union) if union > 0 else 0.0


def align_clusters_to_rice(labels, y_true, k):
    """水田比率が最も高いクラスタを水田クラスタとする"""
    rice_ratio = np.array(
        [y_true[labels == c].mean() if (labels == c).any() else 0.0
         for c in range(k)]
    )
    best_cluster = int(np.argmax(rice_ratio))
    return (labels == best_cluster).astype(int)


def run_region(name: str, csv_path: str):
    # データ読み込み
    df = pd.read_csv(csv_path)

    # 派生特徴量を追加
    df["dNDVI"] = df["NDVI_grow"] - df["NDVI_flood"]
    df["dVH"]   = df["VH_min"] - df["VH_winter"]
    features = BASE_FEATURES + ["dNDVI", "dVH"]

    # 必要カラム確認
    required = features + [LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [{name}] カラム不足: {missing}")
        return

    df = df.dropna(subset=required)
    X = df[features].values
    y_true = (df[LABEL_COL].values == RICE_CLASS).astype(int)

    # 標準化 → K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # 精度計算
    y_pred = align_clusters_to_rice(cluster_labels, y_true, K)
    acc = accuracy_score(y_true, y_pred)
    iou = compute_iou(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n【{name}】 筆数: {len(df):,}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  IoU      : {iou*100:.2f}%")
    print(f"  混同行列 → TP:{tp:,} FP:{fp:,} FN:{fn:,} TN:{tn:,}")


def main():
    print("=" * 50)
    print(f"最新手法精度測定（k={K}・6特徴量）")
    print("=" * 50)

    for name, path in REGIONS.items():
        run_region(name, path)

    print("\n" + "=" * 50)
    print("完了")


if __name__ == "__main__":
    main()
