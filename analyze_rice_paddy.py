"""
つくばみらい市 水田分類結果の分析スクリプト
- K-Means (k=5) でクラスタリング
- mode==3 を水田として Accuracy / IoU を算出
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_DIR = os.path.expanduser("~/Documents/rice_research/")
FEATURES = ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"]
LABEL_COL = "mode"
RICE_CLASS = 3
K = 5
RANDOM_STATE = 42


def load_data(data_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSVファイルが見つかりません: {data_dir}")
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"読み込み完了: {len(csv_files)} ファイル, {len(df):,} 行")
    return df


def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary IoU (Intersection over Union) for rice paddy class."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    union = tp + fp + fn
    return float(tp / union) if union > 0 else 0.0


def align_clusters_to_rice(
    labels: np.ndarray, y_true_binary: np.ndarray, k: int
) -> np.ndarray:
    """
    各クラスタを水田 or 非水田に割り当てる。
    クラスタ内の水田比率が最も高いクラスタを水田クラスタとする。
    """
    rice_ratio = np.array(
        [y_true_binary[labels == c].mean() if (labels == c).any() else 0.0
         for c in range(k)]
    )
    best_cluster = int(np.argmax(rice_ratio))
    return (labels == best_cluster).astype(int)


def main():
    # --- データ読み込み ---
    df = load_data(DATA_DIR)

    required_cols = FEATURES + [LABEL_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"必要なカラムが不足しています: {missing}")

    df = df.dropna(subset=required_cols)
    print(f"欠損除去後: {len(df):,} 行")

    X = df[FEATURES].values
    y_true = (df[LABEL_COL].values == RICE_CLASS).astype(int)

    rice_count = y_true.sum()
    print(f"\n正解ラベル内訳:")
    print(f"  水田 (mode==3) : {rice_count:,} ({rice_count/len(y_true)*100:.1f}%)")
    print(f"  非水田         : {len(y_true)-rice_count:,}")

    # --- 標準化 ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- K-Means クラスタリング ---
    print(f"\nK-Means (k={K}) を実行中...")
    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # --- 水田クラスタの特定 ---
    y_pred = align_clusters_to_rice(cluster_labels, y_true, K)

    # --- 評価指標 ---
    acc = accuracy_score(y_true, y_pred)
    iou = compute_iou(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n========== 評価結果 ==========")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  IoU      : {iou:.4f} ({iou*100:.2f}%)")
    print(f"\n  混同行列 (非水田=0, 水田=1):")
    print(f"          予測:非水田  予測:水田")
    print(f"  実:非水田   {tn:>7,}    {fp:>7,}")
    print(f"  実:水田     {fn:>7,}    {tp:>7,}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"\n  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print("================================")

    # --- クラスタ別統計 ---
    print("\n--- クラスタ別 水田比率 ---")
    df_result = df.copy()
    df_result["cluster"] = cluster_labels
    df_result["y_true"] = y_true
    for c in range(K):
        mask = cluster_labels == c
        ratio = y_true[mask].mean() if mask.any() else 0.0
        marker = " <-- 水田クラスタ" if (mask & (y_pred == 1)).any() else ""
        print(f"  cluster {c}: {mask.sum():>6,} サンプル, 水田比率 {ratio:.3f}{marker}")


if __name__ == "__main__":
    main()
