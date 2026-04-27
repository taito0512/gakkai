"""
v3特徴量（EVI・NDMI・GLCMテクスチャ）の効果をつくばみらい単体で素早く確認する。

評価方法: StratifiedKFold 5分割CV（全3地域が揃うまでの暫定評価）
比較:
  A) v2特徴量（8個・現行 ALL_FEATURES）
  B) v3特徴量（v2 + EVI_mean + NDMI_mean + VH_contrast/ent/idm/corr = 14個）

使い方:
  python scripts/quick_v3_test.py
"""

import copy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---- 設定 ----
LABELS_CSV   = "labels/combined_labels.csv"
FEATURE_CSV  = "data/tsukubamirai_features.csv"
REGION       = "つくばみらい市"

_GDRIVE = "/Users/nagatomotaito/Library/CloudStorage/GoogleDrive-ntaito.50@gmail.com/マイドライブ"
V2_CSV = f"{_GDRIVE}/tsukubamirai_features_v2_2023.csv"
V3_CSV = f"{_GDRIVE}/tsukubamirai_features_v3_2023.csv"

BASE_FEATURES = ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"]
V2_TS_COLS    = ["NDVI_mean", "NDVI_max", "NDVI_min", "NDVI_range", "NDVI_std",
                 "NDWI_mean", "NDVI_apr", "NDVI_jun", "NDVI_aug", "NDVI_oct",
                 "VH_mean", "VH_std"]
V3_ADD_COLS   = ["EVI_mean", "NDMI_mean", "VH_contrast", "VH_ent", "VH_idm", "VH_corr"]

# 現行 ALL_FEATURES（SHAP上位8）
FEAT_V2 = ["NDVI_apr", "NDVI_flood", "dNDVI", "dVH",
           "NDWI_mean", "VH_min", "NDVI_mean", "area_m2"]
# v3追加込み（SHAP上位8 + 新規6）
FEAT_V3 = FEAT_V2 + V3_ADD_COLS

COORD_THR    = 0.001
RANDOM_STATE = 42
N_SPLITS     = 5


def classify_memo(memo) -> str:
    if pd.isna(memo):
        return "不明"
    memo = str(memo)
    if any(k in memo for k in ["水田", "緑色管理圃場", "湛水", "畦畔", "冠水", "収穫後水田"]):
        return "水田っぽい"
    if any(k in memo for k in ["褐色耕起", "耕起", "畝列", "列作物", "トラクター",
                                "施設園芸", "畑", "マルチ", "ハウス", "均一な列",
                                "管理明確", "果樹", "緑色作物", "緑の作物", "褐色農地",
                                "耕起地", "耕起裸地", "耕起痕", "区画境界", "茶色耕起"]):
        return "畑っぽい"
    return "不明"


def compute_iou(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    union = tp + fp + fn
    return float(tp / union) if union > 0 else 0.0


def load_dataset(ts_csv, feat_cols, include_rice=True):
    """ラベルCSV + 特徴量CSV + 時系列CSVを統合して返す"""
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    labels.rename(columns={"面積m2": "area_m2"}, inplace=True)
    labels = labels[labels["地域"] == REGION]

    abandoned = labels[labels["ラベル"] == "放棄地"].copy()
    abandoned["クラス"] = 1

    working = labels[labels["ラベル"] == "耕作中"].copy()
    working["memo_type"] = working["メモ"].apply(classify_memo)
    hatake = working[working["memo_type"] == "畑っぽい"].copy()
    hatake["クラス"] = 0

    combined = pd.concat([abandoned, hatake], ignore_index=True)

    # 基本特徴量CSV
    feat_df = pd.read_csv(FEATURE_CSV, low_memory=False)
    feat_df["dNDVI"] = feat_df["NDVI_grow"] - feat_df["NDVI_flood"]
    feat_df["dVH"]   = feat_df["VH_min"]    - feat_df["VH_winter"]

    # 時系列特徴量をJOIN
    ts = pd.read_csv(ts_csv)
    feat_df = feat_df.merge(ts, on="polygon_uu", how="left")

    # ラベルと最近傍マッチング
    tree = cKDTree(feat_df[["point_lat", "point_lng"]].values)
    dists, idxs = tree.query(combined[["lat", "lon"]].values, k=1)
    result = combined.copy()
    copy_cols = [c for c in feat_cols if c in feat_df.columns and c != "area_m2"]
    for col in copy_cols:
        result[col] = np.where(dists < COORD_THR, feat_df.iloc[idxs][col].values, np.nan)

    # K-Means高確信度水田を疑似ラベルとして追加（3値分類用）
    if include_rice:
        df_clean = feat_df.dropna(subset=BASE_FEATURES).copy()
        df_clean["dNDVI"] = df_clean.get("dNDVI", df_clean["NDVI_grow"] - df_clean["NDVI_flood"])
        df_clean["dVH"]   = df_clean.get("dVH",   df_clean["VH_min"]    - df_clean["VH_winter"])
        X4 = df_clean[BASE_FEATURES].values
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X4)
        km = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_sc)
        vh_means = np.array([X4[km.labels_ == c, 0].mean()
                             if (km.labels_ == c).any() else np.inf for c in range(4)])
        rice_cluster = int(np.argmin(vh_means))
        rice_idx = np.where(km.labels_ == rice_cluster)[0]
        center = km.cluster_centers_[rice_cluster]
        dists_r = np.linalg.norm(X_sc[rice_idx] - center, axis=1)
        hc_idx = rice_idx[dists_r <= np.percentile(dists_r, 33)]
        rng = np.random.default_rng(RANDOM_STATE)
        sampled = rng.choice(hc_idx, size=min(100, len(hc_idx)), replace=False)
        rice_df = df_clean.iloc[sampled].copy()
        rice_df["クラス"] = 2
        rice_df["地域"]   = REGION
        if "area_m2" not in rice_df.columns:
            rice_df["area_m2"] = 1000.0
        else:
            rice_df["area_m2"] = pd.to_numeric(rice_df["area_m2"], errors="coerce").fillna(1000.0)
        result = pd.concat([result, rice_df], ignore_index=True)

    result["area_m2"] = pd.to_numeric(result["area_m2"], errors="coerce").fillna(686.0)
    result = result.dropna(subset=feat_cols)
    return result


def run_cv(dataset, feat_cols, label=""):
    """StratifiedKFold 5分割CVで放棄地F1・IoUを返す"""
    X = dataset[feat_cols].values
    y = dataset["クラス"].values

    models = {
        "LR": Pipeline([("s", StandardScaler()),
                        ("c", LogisticRegression(class_weight="balanced",
                                                  max_iter=2000, random_state=RANDOM_STATE))]),
        "RF": Pipeline([("s", StandardScaler()),
                        ("c", RandomForestClassifier(class_weight="balanced",
                                                      n_estimators=200, random_state=RANDOM_STATE,
                                                      n_jobs=-1))]),
        "GBM": Pipeline([("s", StandardScaler()),
                         ("c", GradientBoostingClassifier(n_estimators=200,
                                                           random_state=RANDOM_STATE))]),
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print(f"\n  [{label}]  特徴量{len(feat_cols)}個  サンプル{len(dataset)}件")

    results = {}
    for name, pipe in models.items():
        f1s, ious = [], []
        for tr, te in skf.split(X, y):
            p = copy.deepcopy(pipe)
            p.fit(X[tr], y[tr])
            yp = p.predict(X[te])
            yt = y[te]
            f1s.append(f1_score(yt == 1, yp == 1, zero_division=0))
            ious.append(compute_iou(yt == 1, yp == 1))
        results[name] = (np.mean(f1s), np.mean(ious))
        print(f"    {name:<5}  F1(放棄地)={np.mean(f1s)*100:.2f}%  IoU={np.mean(ious)*100:.2f}%")
    return results


def main():
    print("=" * 55)
    print("v3特徴量 効果検証（つくばみらい市・5-fold CV）")
    print("=" * 55)

    print("\n■ v2特徴量（8個）でデータ構築中...")
    ds_v2 = load_dataset(V2_CSV, FEAT_V2)
    r_v2 = run_cv(ds_v2, FEAT_V2, label="v2・8特徴量")

    print("\n■ v3特徴量（14個）でデータ構築中...")
    ds_v3 = load_dataset(V3_CSV, FEAT_V3)
    r_v3 = run_cv(ds_v3, FEAT_V3, label="v3・14特徴量")

    # 差分サマリー
    print("\n" + "=" * 55)
    print("比較サマリー（v3 - v2）")
    print("=" * 55)
    print(f"  {'モデル':<6}  {'v2 F1':>8}  {'v3 F1':>8}  {'ΔF1':>7}  {'v2 IoU':>8}  {'v3 IoU':>8}")
    print("  " + "─" * 55)
    for name in r_v2:
        f2, i2 = r_v2[name]
        f3, i3 = r_v3[name]
        df = (f3 - f2) * 100
        sign = "+" if df >= 0 else ""
        print(f"  {name:<6}  {f2*100:>7.2f}%  {f3*100:>7.2f}%  {sign}{df:>5.2f}%"
              f"  {i2*100:>7.2f}%  {i3*100:>7.2f}%")


if __name__ == "__main__":
    main()
