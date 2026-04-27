"""
香取市単体 5-fold CV 精度確認

使用データ:
  - labels/combined_labels.csv（香取市分：放棄地35件・耕作中90件）
  - GDrive/katori_features_base_2023.csv（VH_min, NDVI_flood, NDVI_grow, VH_winter）
  - GDrive/katori_features_v2_2023.csv（NDVI_mean, NDWI_mean, NDVI_apr 等）

使い方:
  python scripts/quick_katori_cv.py
"""

import copy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---- 設定 ----
LABELS_CSV = "labels/combined_labels.csv"
REGION     = "香取市"

_GDRIVE   = (
    "/Users/nagatomotaito/Library/CloudStorage"
    "/GoogleDrive-ntaito.50@gmail.com/マイドライブ"
)
BASE_CSV  = f"{_GDRIVE}/katori_features_base_2023.csv"   # VH_min, NDVI_flood 等
V2_CSV    = f"{_GDRIVE}/katori_features_v2_2023.csv"      # NDVI_mean, NDVI_apr 等

# 学習に使う特徴量（他3地域と同じ8特徴量）
ALL_FEATURES = [
    "NDVI_apr", "NDVI_flood", "dNDVI", "dVH",
    "NDWI_mean", "VH_min", "NDVI_mean", "area_m2",
]

COORD_THR    = 0.001
RANDOM_STATE = 42
N_SPLITS     = 5


def compute_iou(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    union = tp + fp + fn
    return float(tp / union) if union > 0 else 0.0


def load_dataset():
    # ---- ラベル読み込み ----
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    labels.rename(columns={"面積m2": "area_m2"}, inplace=True)
    labels = labels[labels["地域"] == REGION].copy()

    abandoned = labels[labels["ラベル"] == "放棄地"].copy()
    abandoned["クラス"] = 1

    working = labels[labels["ラベル"] == "耕作中"].copy()
    working["クラス"] = 0

    combined = pd.concat([abandoned, working], ignore_index=True)
    print(f"  ラベル数: 放棄地={len(abandoned)}件  耕作中={len(working)}件")

    # ---- 特徴量CSV読み込み・結合 ----
    base = pd.read_csv(BASE_CSV)
    v2   = pd.read_csv(V2_CSV)

    # 派生特徴量
    base["dNDVI"] = base["NDVI_grow"]  - base["NDVI_flood"]
    base["dVH"]   = base["VH_min"]     - base["VH_winter"]

    # base + v2 を polygon_uu で結合
    feat_df = base.merge(v2, on="polygon_uu", how="inner")

    # ---- 最近傍マッチングでラベルに特徴量を付与 ----
    tree = cKDTree(feat_df[["point_lat", "point_lng"]].values)
    dists, idxs = tree.query(combined[["lat", "lon"]].values, k=1)

    copy_cols = [c for c in ALL_FEATURES if c in feat_df.columns and c != "area_m2"]
    for col in copy_cols:
        combined[col] = np.where(
            dists < COORD_THR,
            feat_df.iloc[idxs][col].values,
            np.nan
        )

    # area_m2 はラベルCSVから（なければデフォルト値）
    combined["area_m2"] = pd.to_numeric(combined.get("area_m2", np.nan), errors="coerce").fillna(686.0)

    combined = combined.dropna(subset=ALL_FEATURES)
    print(f"  特徴量マッチ後: {len(combined)}件")
    return combined


def run_cv(dataset):
    X = dataset[ALL_FEATURES].values
    y = dataset["クラス"].values

    models = {
        "LR" : Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(class_weight="balanced",
                                                   max_iter=2000, random_state=RANDOM_STATE))]),
        "RF" : Pipeline([("s", StandardScaler()),
                         ("c", RandomForestClassifier(class_weight="balanced",
                                                       n_estimators=200, random_state=RANDOM_STATE,
                                                       n_jobs=-1))]),
        "GBM": Pipeline([("s", StandardScaler()),
                         ("c", GradientBoostingClassifier(n_estimators=200,
                                                           random_state=RANDOM_STATE))]),
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print(f"\n  5-fold CV  特徴量{len(ALL_FEATURES)}個  サンプル{len(dataset)}件")
    print(f"  {'モデル':<5}  {'F1(放棄地)':>10}  {'IoU':>8}")
    print("  " + "─" * 30)

    for name, pipe in models.items():
        f1s, ious = [], []
        for tr, te in skf.split(X, y):
            p = copy.deepcopy(pipe)
            p.fit(X[tr], y[tr])
            yp = p.predict(X[te])
            yt = y[te]
            f1s.append(f1_score(yt == 1, yp == 1, zero_division=0))
            ious.append(compute_iou(yt == 1, yp == 1))
        print(f"  {name:<5}  {np.mean(f1s)*100:>9.2f}%  {np.mean(ious)*100:>7.2f}%")


def main():
    print("=" * 50)
    print("  香取市 放棄地分類 5-fold CV")
    print("=" * 50)
    ds = load_dataset()
    run_cv(ds)


if __name__ == "__main__":
    main()
