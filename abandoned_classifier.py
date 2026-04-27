"""
第2段階：耕作放棄地分類モデル
- 対象: 非水田農地（第1段階K-Meansで非水田と判定された筆）
- タスク A（2値）: 畑(0) vs 放棄地(1)
- タスク B（3値）: 畑(0) vs 放棄地(1) vs 水田(2)  ← 第1段階の取り逃がしを補正
- モデル: RF / GBM / SVM / LR / LightGBM を比較
- チューニング: RandomizedSearchCV（全データ・標準5-fold）でベストパラメータ探索
- 評価: 地域leave-one-out空間交差検証（空間的汎化性を評価）
- 特徴量: VH_min, NDVI_grow, VH_winter, NDVI_flood, dNDVI, dVH, area_m2
"""

import copy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, make_scorer
)
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# --- 設定 ---
LABELS_CSV = "labels/combined_labels.csv"
_GDRIVE    = "/Users/nagatomotaito/Library/CloudStorage/GoogleDrive-ntaito.50@gmail.com/マイドライブ"

FEATURE_CSVS = {
    "つくばみらい市": "data/tsukubamirai_features.csv",
    "稲敷市":         "data/inashiki_features.csv",
    "笠間市":         "data/kasama_features.csv",
    "香取市":         f"{_GDRIVE}/katori_features_base_2023.csv",
}

# 時系列追加特徴量CSV v2
TIMESERIES_CSVS = {
    "つくばみらい市": f"{_GDRIVE}/tsukubamirai_features_v2_2023.csv",
    "稲敷市":         f"{_GDRIVE}/inashiki_features_v2_2023.csv",
    "笠間市":         f"{_GDRIVE}/kasama_features_v2_2023.csv",
    "香取市":         f"{_GDRIVE}/katori_features_v2_2023.csv",
}
TIMESERIES_COLS = [
    "NDVI_mean", "NDVI_max", "NDVI_min", "NDVI_range", "NDVI_std",
    "NDWI_mean",
    "NDVI_apr", "NDVI_jun", "NDVI_aug", "NDVI_oct",
    "VH_mean", "VH_std",
    # v3追加: EVI・NDMI・GLCMテクスチャ
    "EVI_mean", "NDMI_mean",
    "VH_contrast", "VH_ent", "VH_idm", "VH_corr",
]

BASE_FEATURES = ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"]
# SHAP検証済み: 上位8特徴量（v3追加特徴量は改善なし→v2で十分）
ALL_FEATURES  = [
    "NDVI_apr", "NDVI_flood", "dNDVI", "dVH",
    "NDWI_mean", "VH_min", "NDVI_mean", "area_m2",
]
# アブレーション比較用（v2・8特徴量）
REDUCED_FEATURES = [
    "NDVI_apr", "NDVI_flood", "dNDVI", "dVH",
    "NDWI_mean", "VH_min", "NDVI_mean", "area_m2",
]

COORD_THRESHOLD = 0.001  # 度（約100m）
RANDOM_STATE    = 42
STAGE1_K        = 4      # K-Meansのクラスタ数（第1段階と同じ設定）
N_ITER_SEARCH   = 50     # RandomizedSearchCVの試行回数
CV_FOLDS        = 5      # チューニング時の内側CV


# =====================================================================
# 補助関数
# =====================================================================

def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """放棄地クラス(1)のIoUを計算"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    union = tp + fp + fn
    return float(tp / union) if union > 0 else 0.0


def iou_scorer(y_true, y_pred):
    return compute_iou(np.array(y_true), np.array(y_pred))


def classify_memo(memo) -> str:
    """耕作中メモを 畑っぽい / 水田っぽい / 不明 に分類"""
    if pd.isna(memo):
        return "不明"
    memo = str(memo)
    # 水田っぽいキーワード（先に判定）
    if any(k in memo for k in ["水田", "緑色管理圃場", "湛水", "畦畔", "冠水", "収穫後水田"]):
        return "水田っぽい"
    # 畦は水田と畑どちらにもあるので除外。畦畔は水田に限定
    if any(k in memo for k in [
        "褐色耕起", "耕起", "畝列", "列作物", "トラクター", "施設園芸", "畑",
        "マルチ", "ハウス", "均一な列", "管理明確", "果樹", "緑色作物",
        "緑の作物", "褐色農地", "耕起地", "耕起裸地", "耕起痕",
        "区画境界", "茶色耕起",
    ]):
        return "畑っぽい"
    return "不明"


def load_features(region: str) -> pd.DataFrame:
    """特徴量CSVを読み込み、派生特徴量（dNDVI, dVH）と時系列特徴量を追加"""
    df = pd.read_csv(FEATURE_CSVS[region], low_memory=False)
    df["dNDVI"] = df["NDVI_grow"] - df["NDVI_flood"]
    df["dVH"]   = df["VH_min"]    - df["VH_winter"]

    # 時系列特徴量をpolygon_uuでJOIN（CSVに存在する列だけ読む）
    if region in TIMESERIES_CSVS:
        ts_raw = pd.read_csv(TIMESERIES_CSVS[region])
        available = [c for c in TIMESERIES_COLS if c in ts_raw.columns]
        ts = ts_raw[["polygon_uu"] + available]
        df = df.merge(ts, on="polygon_uu", how="left")

    return df


def match_labels_to_features(labels_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    """教師データ(lat,lon)と特徴量CSV(point_lat,point_lng)を最近傍でマッチング"""
    tree = cKDTree(feat_df[["point_lat", "point_lng"]].values)
    dists, idxs = tree.query(labels_df[["lat", "lon"]].values, k=1)

    result = labels_df.copy()
    copy_cols = BASE_FEATURES + ["dNDVI", "dVH"] + [c for c in TIMESERIES_COLS if c in feat_df.columns]
    for col in copy_cols:
        result[col] = np.where(
            dists < COORD_THRESHOLD,
            feat_df.iloc[idxs][col].values,
            np.nan
        )

    n_miss = (dists >= COORD_THRESHOLD).sum()
    if n_miss > 0:
        print(f"    ⚠ マッチ失敗: {n_miss}/{len(labels_df)}件（閾値 {COORD_THRESHOLD}度超）")
    return result


# =====================================================================
# データ準備
# =====================================================================

def prepare_dataset() -> pd.DataFrame:
    """
    combined_labels.csv を整形・特徴量マッチングして返す。
    クラス0=畑（畑っぽい耕作中）、クラス1=放棄地
    """
    print("=== データ読み込み ===")
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    labels.rename(columns={"面積m2": "area_m2"}, inplace=True)
    print(f"  combined_labels.csv: {len(labels):,}行")

    abandoned = labels[labels["ラベル"] == "放棄地"].copy()
    abandoned["クラス"] = 1

    working = labels[labels["ラベル"] == "耕作中"].copy()
    working["memo_type"] = working["メモ"].apply(classify_memo)
    hatake = working[working["memo_type"] == "畑っぽい"].copy()
    hatake["クラス"] = 0

    print(f"  放棄地: {len(abandoned):,}件  畑っぽい耕作中: {len(hatake):,}件")

    combined = pd.concat([abandoned, hatake], ignore_index=True)
    print("\n  地域別内訳:")
    bd = combined.groupby(["地域", "クラス"]).size().unstack(fill_value=0)
    bd.columns = ["畑(0)", "放棄地(1)"]
    print(bd.to_string())

    all_rows = []
    for region in FEATURE_CSVS:
        subset = combined[combined["地域"] == region].copy()
        if len(subset) == 0:
            continue
        print(f"\n  [{region}] 特徴量マッチング中... ({len(subset)}件)")
        matched = match_labels_to_features(subset, load_features(region))
        all_rows.append(matched)

    dataset = pd.concat(all_rows, ignore_index=True)
    dataset["area_m2"] = pd.to_numeric(dataset["area_m2"], errors="coerce")
    # area_m2が空欄の行（farm_candidate等）はグローバル中央値で補完
    area_median = dataset["area_m2"].median()
    n_filled = dataset["area_m2"].isna().sum()
    if n_filled > 0:
        dataset["area_m2"] = dataset["area_m2"].fillna(area_median)
        print(f"  area_m2補完: {n_filled}件 → 中央値{area_median:.0f}m2で代替")

    before = len(dataset)
    dataset = dataset.dropna(subset=ALL_FEATURES)
    after   = len(dataset)
    if before != after:
        print(f"\n  NaN除去: {before - after}件削除 → {after}件使用")

    return dataset


# =====================================================================
# データ準備（3値分類版）
# =====================================================================

def get_stage1_highconf_rice_samples(n_per_region: int = 100,
                                      random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    各地域でStage1 K-Meansを実行し、高確信度（クラスタ中心距離 下位33%）の
    水田サンプルを n_per_region 件ずつ疑似ラベルとして取得する。

    背景: 目視ラベリングの水田っぽいサンプルは30件しかないため、
    K-Meansが97〜99.9%の適合率で判定した高確信度水田を class=2 として補充する。
    """
    all_samples = []
    rng = np.random.default_rng(random_state)

    for region, feat_path in FEATURE_CSVS.items():
        df = pd.read_csv(feat_path, low_memory=False)
        df["dNDVI"] = df["NDVI_grow"] - df["NDVI_flood"]
        df["dVH"]   = df["VH_min"]    - df["VH_winter"]

        # 時系列特徴量をpolygon_uuでJOIN
        if region in TIMESERIES_CSVS:
            ts_raw = pd.read_csv(TIMESERIES_CSVS[region])
            available = [c for c in TIMESERIES_COLS if c in ts_raw.columns]
            ts = ts_raw[["polygon_uu"] + available]
            df = df.merge(ts, on="polygon_uu", how="left")

        # area_m2が特徴量CSVにない場合は中央値を計算後に追加
        if "area_m2" not in df.columns:
            df["area_m2"] = np.nan  # dropna後に補完するためNaNで初期化

        # 基本特徴量のみでNaN除去（面積はあとで補完）
        df_clean = df.dropna(subset=BASE_FEATURES + ["dNDVI", "dVH"]).reset_index(drop=True)

        # 面積の補完（NaNなら地域平均1000m2で代替）
        df_clean["area_m2"] = pd.to_numeric(df_clean["area_m2"], errors="coerce").fillna(1000.0)

        # K-Means（4クラスタ・基本4特徴量）
        X4 = df_clean[BASE_FEATURES].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X4)
        km = KMeans(n_clusters=STAGE1_K, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        cluster_labels = km.labels_

        # VH_min(index=0)が最小のクラスタ → 水田
        vh_means = np.array([
            X4[cluster_labels == c, 0].mean() if (cluster_labels == c).any() else np.inf
            for c in range(STAGE1_K)
        ])
        rice_cluster = int(np.argmin(vh_means))
        rice_idx = np.where(cluster_labels == rice_cluster)[0]

        # クラスタ中心からの距離を計算し、下位33%を高確信度とする
        center = km.cluster_centers_[rice_cluster]
        dists  = np.linalg.norm(X_scaled[rice_idx] - center, axis=1)
        threshold = np.percentile(dists, 33)
        high_conf_idx = rice_idx[dists <= threshold]

        # n_per_region 件ランダムサンプリング
        if len(high_conf_idx) > n_per_region:
            sampled = rng.choice(high_conf_idx, size=n_per_region, replace=False)
        else:
            sampled = high_conf_idx

        samples = df_clean.iloc[sampled].copy()
        samples["地域"]  = region
        samples["クラス"] = 2
        # 面積列がない場合は中央値で補完
        if "area_m2" not in samples.columns:
            samples["area_m2"] = 1000.0
        else:
            samples["area_m2"] = pd.to_numeric(samples["area_m2"], errors="coerce").fillna(1000.0)

        all_samples.append(samples)
        print(f"    [{region}] 高確信度水田: {len(high_conf_idx):,}件中 {len(sampled)}件を疑似ラベルに使用")

    return pd.concat(all_samples, ignore_index=True)


def prepare_dataset_3class(n_rice_pseudo: int = 100) -> pd.DataFrame:
    """
    3値分類用データセットを構築する。
    クラス: 0=畑, 1=放棄地, 2=水田

    水田サンプルは2種類を混合:
      A) 目視ラベリング「水田っぽい」（30件・本物ラベル）
      B) K-Means高確信度水田（地域ごと n_rice_pseudo 件・疑似ラベル）
    """
    print("=== データ読み込み（3値分類・K-Means疑似ラベル水田補充）===")
    labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    labels.rename(columns={"面積m2": "area_m2"}, inplace=True)

    abandoned = labels[labels["ラベル"] == "放棄地"].copy()
    abandoned["クラス"] = 1

    working = labels[labels["ラベル"] == "耕作中"].copy()
    working["memo_type"] = working["メモ"].apply(classify_memo)
    hatake = working[working["memo_type"] == "畑っぽい"].copy()
    hatake["クラス"] = 0
    suiden_label = working[working["memo_type"] == "水田っぽい"].copy()
    suiden_label["クラス"] = 2

    # ラベル由来サンプルに特徴量をマッチング
    label_combined = pd.concat([abandoned, hatake, suiden_label], ignore_index=True)
    all_rows = []
    for region in FEATURE_CSVS:
        subset = label_combined[label_combined["地域"] == region].copy()
        if len(subset) == 0:
            continue
        matched = match_labels_to_features(subset, load_features(region))
        all_rows.append(matched)

    label_dataset = pd.concat(all_rows, ignore_index=True)
    label_dataset["area_m2"] = pd.to_numeric(label_dataset["area_m2"], errors="coerce")
    label_dataset = label_dataset.dropna(subset=ALL_FEATURES)

    print(f"\n  ラベル由来: 放棄地{(label_dataset['クラス']==1).sum()} / "
          f"畑{(label_dataset['クラス']==0).sum()} / "
          f"水田(ラベル){(label_dataset['クラス']==2).sum()}")

    # K-Means疑似ラベル水田を追加
    print(f"\n  K-Means高確信度水田サンプル取得中（地域ごと最大{n_rice_pseudo}件）...")
    pseudo_rice = get_stage1_highconf_rice_samples(n_per_region=n_rice_pseudo)
    pseudo_rice = pseudo_rice.dropna(subset=ALL_FEATURES)
    print(f"  疑似ラベル水田: 計{len(pseudo_rice)}件")

    dataset = pd.concat([label_dataset, pseudo_rice], ignore_index=True)

    print(f"\n  地域別内訳（合計）:")
    bd = dataset.groupby(["地域", "クラス"]).size().unstack(fill_value=0)
    bd = bd.rename(columns={0: "畑(0)", 1: "放棄地(1)", 2: "水田(2)"})
    print(bd.to_string())
    print(f"\n  → 合計: {len(dataset)}件  "
          f"（放棄地{(dataset['クラス']==1).sum()} / "
          f"畑{(dataset['クラス']==0).sum()} / "
          f"水田{(dataset['クラス']==2).sum()}）")

    return dataset


# =====================================================================
# モデル定義 + ハイパーパラメータ探索空間
# =====================================================================

def get_model_configs() -> list[dict]:
    """
    (モデル名, 初期モデル, パラメータグリッド) のリストを返す。
    Pipelineで StandardScaler を先に挟み、パラメータ名は clf__ プレフィックス。
    """
    configs = [
        {
            "name": "RandomForest",
            "clf": RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            "params": {
                "clf__n_estimators":  [100, 200, 300, 500],
                "clf__max_depth":     [None, 5, 10, 20],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features":  ["sqrt", "log2", 0.5],
            },
        },
        {
            "name": "GradientBoosting",
            "clf": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "clf__n_estimators":  [100, 200, 300],
                "clf__max_depth":     [3, 4, 5, 6],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__subsample":     [0.6, 0.8, 1.0],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        },
        {
            "name": "SVM",
            "clf": SVC(class_weight="balanced", random_state=RANDOM_STATE),
            "params": {
                "clf__C":      [0.1, 1, 10, 100],
                "clf__kernel": ["rbf", "linear"],
                "clf__gamma":  ["scale", "auto", 0.01, 0.1],
            },
        },
        {
            "name": "LogisticRegression",
            "clf": LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            "params": {
                "clf__C":      [0.01, 0.1, 1, 10, 100],
                "clf__penalty": ["l1", "l2"],
                "clf__solver":  ["liblinear", "saga"],
            },
        },
    ]
    if HAS_LGBM:
        configs.append({
            "name": "LightGBM",
            "clf": LGBMClassifier(
                class_weight="balanced", random_state=RANDOM_STATE,
                n_jobs=-1, verbose=-1),
            "params": {
                "clf__n_estimators":  [100, 200, 300, 500],
                "clf__max_depth":     [-1, 4, 6, 8],
                "clf__learning_rate": [0.01, 0.05, 0.1],
                "clf__num_leaves":    [15, 31, 63],
                "clf__min_child_samples": [5, 10, 20],
                "clf__subsample":     [0.7, 0.8, 1.0],
                "clf__colsample_bytree": [0.7, 0.8, 1.0],
            },
        })
    return configs


def build_ensemble_pipeline(best_pipelines: dict[str, Pipeline]) -> VotingClassifier:
    """
    チューニング済みの GBM・RF・LightGBM を soft voting でまとめる。
    各サブモデルはすでに StandardScaler 込みの Pipeline なので
    VotingClassifier にそのまま渡す（二重スケーリング不要）。
    戻り値は Pipeline ではなく VotingClassifier だが、
    fit/predict/predict_proba インターフェースは共通なので run_spatial_cv で使える。
    """
    estimators = []
    for name in ["GradientBoosting", "RandomForest", "LightGBM"]:
        if name in best_pipelines:
            estimators.append((name, copy.deepcopy(best_pipelines[name])))

    if len(estimators) < 2:
        raise ValueError("アンサンブルには最低2モデル必要")

    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


# =====================================================================
# ハイパーパラメータチューニング
# =====================================================================

def tune_models(dataset: pd.DataFrame) -> dict[str, Pipeline]:
    """
    全データに対してRandomizedSearchCVでチューニングし、
    ベストパイプライン（StandardScaler + チューニング済みモデル）を返す。
    チューニング指標: F1（放棄地クラス）
    """
    print("\n=== ハイパーパラメータチューニング（RandomizedSearchCV, 5-fold）===")
    X = dataset[ALL_FEATURES].values
    y = dataset["クラス"].values

    scorer = make_scorer(f1_score, zero_division=0)
    best_pipelines = {}

    for cfg in get_model_configs():
        name   = cfg["name"]
        pipe   = Pipeline([("scaler", StandardScaler()), ("clf", cfg["clf"])])
        search = RandomizedSearchCV(
            pipe,
            param_distributions=cfg["params"],
            n_iter=N_ITER_SEARCH,
            scoring=scorer,
            cv=CV_FOLDS,
            refit=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X, y)
        best_pipelines[name] = search.best_estimator_

        best_params = {k.replace("clf__", ""): v
                       for k, v in search.best_params_.items()}
        print(f"\n  [{name}]  best CV-F1: {search.best_score_*100:.2f}%")
        print(f"    → {best_params}")

    return best_pipelines


# =====================================================================
# Leave-One-Region-Out 空間交差検証
# =====================================================================

def run_spatial_cv(dataset: pd.DataFrame,
                   pipelines: dict[str, Pipeline]) -> list[dict]:
    """
    チューニング済みパイプラインを使い、3地域 leave-one-out 空間CVを実行。
    各foldで scaler + モデルをゼロから再訓練（データリーク防止）。
    """
    regions     = list(FEATURE_CSVS.keys())
    all_results = []

    print("\n=== Leave-One-Region-Out 空間交差検証（チューニング済みパラメータ）===")

    for name, best_pipe in pipelines.items():
        print(f"\n{'─'*55}")
        print(f"■ {name}")
        all_y_true, all_y_pred = [], []

        for test_region in regions:
            train_df = dataset[dataset["地域"] != test_region]
            test_df  = dataset[dataset["地域"] == test_region]
            if len(test_df) == 0:
                continue

            X_train = train_df[ALL_FEATURES].values
            y_train = train_df["クラス"].values
            X_test  = test_df[ALL_FEATURES].values
            y_test  = test_df["クラス"].values

            # ベストパラメータを引き継いだ新しいパイプラインを再訓練
            pipe_copy = copy.deepcopy(best_pipe)
            pipe_copy.fit(X_train, y_train)
            y_pred = pipe_copy.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, zero_division=0)
            iou = compute_iou(y_test, y_pred)
            print(f"  [{test_region}]  Acc:{acc*100:.1f}%  F1:{f1*100:.1f}%  IoU:{iou*100:.1f}%"
                  f"  (n={len(test_df)})")

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

        # 全fold合算
        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)
        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        iou  = compute_iou(y_true, y_pred)
        cm   = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"  [全体合算]  Acc:{acc*100:.2f}%  F1:{f1*100:.2f}%  "
              f"Prec:{prec*100:.2f}%  Rec:{rec*100:.2f}%  IoU:{iou*100:.2f}%")
        print(f"             TP:{tp} FP:{fp} FN:{fn} TN:{tn}")
        all_results.append(dict(
            model=name, acc=acc, f1=f1, prec=prec, rec=rec, iou=iou
        ))

    return all_results


# =====================================================================
# 3値分類 空間交差検証
# =====================================================================

def run_spatial_cv_3class(dataset3: pd.DataFrame,
                           best_pipelines: dict[str, Pipeline]) -> list[dict]:
    """
    3値分類（畑/放棄地/水田）のleave-one-region-out空間CVを実行。
    2値分類との比較のため放棄地クラス(1)のF1・IoUも個別出力する。
    """
    regions     = list(FEATURE_CSVS.keys())
    all_results = []
    CLASS_NAMES = {0: "畑", 1: "放棄地", 2: "水田"}

    print("\n=== 3値分類 Leave-One-Region-Out 空間CV ===")
    print("  クラス: 0=畑 / 1=放棄地 / 2=水田（第1段階取り逃がし補正）")

    for name, best_pipe in best_pipelines.items():
        print(f"\n{'─'*55}")
        print(f"■ {name}")
        all_y_true, all_y_pred = [], []

        for test_region in regions:
            train_df = dataset3[dataset3["地域"] != test_region]
            test_df  = dataset3[dataset3["地域"] == test_region]
            if len(test_df) == 0:
                continue

            # 水田サンプルが訓練にゼロの場合はスキップ警告
            n_suiden_train = (train_df["クラス"] == 2).sum()
            if n_suiden_train == 0:
                print(f"  [{test_region}] ⚠ 訓練データに水田サンプルなし → スキップ")
                continue

            pipe_copy = copy.deepcopy(best_pipe)
            pipe_copy.fit(train_df[ALL_FEATURES], train_df["クラス"].values)
            y_pred = pipe_copy.predict(test_df[ALL_FEATURES])
            y_test = test_df["クラス"].values

            acc     = accuracy_score(y_test, y_pred)
            f1_aban = f1_score(y_test, y_pred, labels=[1], average="macro", zero_division=0)
            f1_mac  = f1_score(y_test, y_pred, average="macro", zero_division=0)
            print(f"  [{test_region}]  Acc:{acc*100:.1f}%  "
                  f"F1(放棄地):{f1_aban*100:.1f}%  F1(macro):{f1_mac*100:.1f}%"
                  f"  (n={len(test_df)})")

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

        if not all_y_true:
            continue

        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)
        acc     = accuracy_score(y_true, y_pred)
        f1_mac  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_aban = f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
        iou_aban = compute_iou(y_true == 1, y_pred == 1)

        # 水田補正効果: 予測クラス2（水田）の中で本当に水田だった割合
        pred_suiden_mask = y_pred == 2
        true_suiden_mask = y_true == 2
        n_pred_suiden = pred_suiden_mask.sum()
        n_correct_suiden = ((y_pred == 2) & (y_true == 2)).sum()
        suiden_prec = n_correct_suiden / n_pred_suiden if n_pred_suiden > 0 else 0.0
        suiden_rec  = n_correct_suiden / true_suiden_mask.sum() if true_suiden_mask.sum() > 0 else 0.0

        print(f"  [全体合算]  Acc:{acc*100:.2f}%  F1(放棄地):{f1_aban*100:.2f}%  "
              f"IoU(放棄地):{iou_aban*100:.2f}%  F1(macro):{f1_mac*100:.2f}%")
        print(f"  [水田補正]  水田予測数:{n_pred_suiden}  "
              f"Precision:{suiden_prec*100:.1f}%  Recall:{suiden_rec*100:.1f}%")
        print(f"\n  詳細レポート（全地域合算）:")
        print(classification_report(
            y_true, y_pred,
            target_names=["畑(0)", "放棄地(1)", "水田(2)"],
            zero_division=0
        ))

        all_results.append(dict(
            model=name, acc=acc, f1_mac=f1_mac,
            f1_aban=f1_aban, iou_aban=iou_aban,
            suiden_prec=suiden_prec, suiden_rec=suiden_rec
        ))

    return all_results


# =====================================================================
# ① 確信度スコアの精度検証
# =====================================================================

def analyze_confidence_precision(dataset: pd.DataFrame,
                                  best_pipeline) -> None:
    """
    GBMの空間CVで得た放棄地確率を確信度3段階に分け、
    各確信度レベルのPrecision（どれだけ本当に放棄地か）を出力する。
    K-Meansの高確信度検証（高確信度30件→30/30全部水田）と同じ考え方。
    """
    print("\n=== ① 確信度スコア精度検証（空間CV・GBM）===")
    regions     = list(FEATURE_CSVS.keys())
    all_y_true  = []
    all_y_prob  = []

    for test_region in regions:
        train_df = dataset[dataset["地域"] != test_region]
        test_df  = dataset[dataset["地域"] == test_region]
        if len(test_df) == 0:
            continue

        pipe_copy = copy.deepcopy(best_pipeline)
        pipe_copy.fit(train_df[ALL_FEATURES].values, train_df["クラス"].values)

        # predict_probaで放棄地(1)の確率を取得
        probs  = pipe_copy.predict_proba(test_df[ALL_FEATURES].values)[:, 1]
        y_test = test_df["クラス"].values

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(probs.tolist())

    y_true = np.array(all_y_true)
    y_prob = np.array(all_y_prob)

    # 放棄地と予測した筆（確率 > 0.5）の中でパーセンタイルで3段階に分ける
    pred_aban_mask = y_prob > 0.5
    if pred_aban_mask.sum() == 0:
        print("  放棄地予測なし → スキップ")
        return

    aban_probs = y_prob[pred_aban_mask]
    p33 = np.percentile(aban_probs, 33)
    p66 = np.percentile(aban_probs, 66)

    levels = {
        "🟢 高確信度（上位33%）": (y_prob >= p66) & pred_aban_mask,
        "🟡 中確信度（中位33%）": (y_prob >= p33) & (y_prob < p66) & pred_aban_mask,
        "🔴 低確信度（下位33%）": (y_prob < p33)  & pred_aban_mask,
    }

    print(f"  {'確信度レベル':<22} {'件数':>6}  {'Precision':>10}  {'うち本物放棄地':>14}")
    print("  " + "─" * 58)
    for label, mask in levels.items():
        n      = mask.sum()
        n_true = (y_true[mask] == 1).sum()
        prec   = n_true / n if n > 0 else 0.0
        print(f"  {label:<22} {n:>6,}件  {prec*100:>9.1f}%  {n_true:>6,}/{n:<6,}")

    # 全体（閾値0.5）の参考
    n_all  = pred_aban_mask.sum()
    n_true_all = (y_true[pred_aban_mask] == 1).sum()
    print(f"  {'合計（閾値0.5）':<22} {n_all:>6,}件  {n_true_all/n_all*100:>9.1f}%  "
          f"{n_true_all:>6,}/{n_all:<6,}")
    print(f"\n  → 高確信度のPrecisionが全体より高いほど確信度スコアが有効")


# =====================================================================
# ② 分類閾値チューニング
# =====================================================================

def analyze_threshold(dataset: pd.DataFrame, best_pipeline) -> None:
    """
    GBM の分類閾値（デフォルト0.5）を変えたときの
    Precision / Recall / F1 の変化を出力する。
    用途に応じた最適閾値を選ぶための参考情報。
    """
    print("\n=== ② 分類閾値チューニング（空間CV・GBM）===")
    regions    = list(FEATURE_CSVS.keys())
    all_y_true = []
    all_y_prob = []

    for test_region in regions:
        train_df = dataset[dataset["地域"] != test_region]
        test_df  = dataset[dataset["地域"] == test_region]
        if len(test_df) == 0:
            continue

        pipe_copy = copy.deepcopy(best_pipeline)
        pipe_copy.fit(train_df[ALL_FEATURES].values, train_df["クラス"].values)
        probs  = pipe_copy.predict_proba(test_df[ALL_FEATURES].values)[:, 1]
        all_y_true.extend(test_df["クラス"].values.tolist())
        all_y_prob.extend(probs.tolist())

    y_true = np.array(all_y_true)
    y_prob = np.array(all_y_prob)

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    print(f"\n  {'閾値':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  "
          f"{'IoU':>8}  {'放棄地予測数':>12}  用途目安")
    print("  " + "─" * 72)

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        iou  = compute_iou(y_true, y_pred)
        n    = y_pred.sum()

        note = ""
        if thr == 0.50:
            note = "← デフォルト"
        elif thr <= 0.40:
            note = "← 見逃し重視（行政用）"
        elif thr >= 0.65:
            note = "← 誤検知削減（現地調査用）"

        marker = "▶" if thr == 0.50 else " "
        print(f"  {marker}{thr:.2f}   {prec*100:>9.1f}%  {rec*100:>7.1f}%  "
              f"{f1*100:>7.1f}%  {iou*100:>7.1f}%  {n:>11,}件  {note}")

    print(f"\n  推奨:")
    print(f"    行政パトロール（見逃し最小化） → 閾値 0.35〜0.40")
    print(f"    現地調査（効率優先）           → 閾値 0.60〜0.65")


# =====================================================================
# 特徴量重要度（ベストモデルで全データ訓練）
# =====================================================================

def analyze_feature_importance(dataset: pd.DataFrame,
                                best_pipelines: dict[str, Pipeline]):
    """RF と LightGBM の特徴量重要度を出力（tree-based モデルのみ）"""
    print("\n=== 特徴量重要度（全データ訓練・参考） ===")
    X = dataset[ALL_FEATURES].values
    y = dataset["クラス"].values

    tree_models = ["RandomForest", "GradientBoosting", "LightGBM"]
    for name in tree_models:
        if name not in best_pipelines:
            continue
        pipe = copy.deepcopy(best_pipelines[name])
        pipe.fit(X, y)
        clf = pipe.named_steps["clf"]
        imp = pd.Series(clf.feature_importances_, index=ALL_FEATURES)
        imp = imp.sort_values(ascending=False)
        print(f"\n  [{name}]")
        for feat, v in imp.items():
            bar = "█" * int(v * 50)
            print(f"    {feat:<12} {v:.4f}  {bar}")


# =====================================================================
# 特徴量絞り込み比較（アブレーション）
# =====================================================================

def run_ablation_comparison(dataset3: pd.DataFrame,
                             best_pipelines: dict) -> None:
    """
    全特徴量(19) vs SHAP上位8特徴量で LR・RF の3値空間CVを比較し、
    特徴量を絞っても精度が維持されるかを確認する。
    """
    print("\n=== 特徴量アブレーション比較（全19 vs SHAP上位8）===")
    print(f"  全特徴量  ({len(ALL_FEATURES):2d}個): {ALL_FEATURES}")
    print(f"  絞込特徴量 ({len(REDUCED_FEATURES):2d}個): {REDUCED_FEATURES}")

    regions = list(FEATURE_CSVS.keys())
    CLASS_NAMES = ["畑(0)", "放棄地(1)", "水田(2)"]

    rows = []
    for feat_name, feat_cols in [(f"SHAP上位{len(ALL_FEATURES)}特徴量（現行）", ALL_FEATURES),
                                   (f"全{len(REDUCED_FEATURES)}特徴量（参考）", REDUCED_FEATURES)]:
        for model_name in ["LogisticRegression", "RandomForest"]:
            if model_name not in best_pipelines:
                continue

            all_y_true, all_y_pred = [], []
            for test_region in regions:
                train_df = dataset3[dataset3["地域"] != test_region]
                test_df  = dataset3[dataset3["地域"] == test_region]
                if len(test_df) == 0:
                    continue
                if (train_df["クラス"] == 2).sum() == 0:
                    continue

                pipe = copy.deepcopy(best_pipelines[model_name])
                pipe.fit(train_df[feat_cols], train_df["クラス"].values)
                y_pred = pipe.predict(test_df[feat_cols])
                all_y_true.extend(test_df["クラス"].values.tolist())
                all_y_pred.extend(y_pred.tolist())

            if not all_y_true:
                continue

            y_true = np.array(all_y_true)
            y_pred = np.array(all_y_pred)
            f1_aban = f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
            iou_aban = compute_iou(y_true == 1, y_pred == 1)
            f1_mac   = f1_score(y_true, y_pred, average="macro", zero_division=0)
            acc      = accuracy_score(y_true, y_pred)

            rows.append({
                "特徴量セット": feat_name,
                "モデル":       model_name,
                "Acc":          acc,
                "F1(放棄地)":   f1_aban,
                "IoU(放棄地)":  iou_aban,
                "F1(macro)":    f1_mac,
            })

    # 表形式で出力
    print(f"\n  {'特徴量セット':<18} {'モデル':<20} {'Acc':>7} {'F1(放棄地)':>11} {'IoU(放棄地)':>12} {'F1(macro)':>10}")
    print("  " + "─" * 75)
    prev_feat = None
    for r in rows:
        sep = "\n" if (prev_feat and prev_feat != r["特徴量セット"]) else ""
        print(f"{sep}  {r['特徴量セット']:<18} {r['モデル']:<20}"
              f" {r['Acc']*100:>6.2f}%"
              f" {r['F1(放棄地)']*100:>10.2f}%"
              f" {r['IoU(放棄地)']*100:>11.2f}%"
              f" {r['F1(macro)']*100:>9.2f}%")
        prev_feat = r["特徴量セット"]

    # 差分（絞込 - 全体）
    print(f"\n  {'── 差分（SHAP上位8 - 全19）':}")
    full_map    = {r["モデル"]: r for r in rows if "現行" in r["特徴量セット"]}
    reduced_map = {r["モデル"]: r for r in rows if "参考" in r["特徴量セット"]}
    for model_name in full_map:
        if model_name not in reduced_map:
            continue
        rf = full_map[model_name]
        rr = reduced_map[model_name]
        df1 = (rr["F1(放棄地)"] - rf["F1(放棄地)"]) * 100
        sign = "+" if df1 >= 0 else ""
        print(f"  {model_name:<20}  ΔF1(放棄地): {sign}{df1:.2f}%  "
              f"ΔIoU: {sign}{(rr['IoU(放棄地)']-rf['IoU(放棄地)'])*100:.2f}%")


# =====================================================================
# SHAP値可視化
# =====================================================================

def analyze_shap(dataset3: pd.DataFrame, best_pipelines: dict) -> None:
    """
    LR（最高精度）のSHAP値を全データで計算し、Beeswarm・Bar plotを保存する。
    放棄地クラス(1)に着目して特徴量寄与を可視化する。
    """
    import shap

    OUT_DIR = Path("figures")
    OUT_DIR.mkdir(exist_ok=True)

    print("\n=== SHAP値可視化（LogisticRegression・3値分類）===")

    if "LogisticRegression" not in best_pipelines:
        print("  LogisticRegression が見つかりません → スキップ")
        return

    # 全データで再訓練
    pipe = copy.deepcopy(best_pipelines["LogisticRegression"])
    X = dataset3[ALL_FEATURES].values
    y = dataset3["クラス"].values
    pipe.fit(X, y)

    # Pipelineからscalerとモデルを取り出す
    scaler = pipe.named_steps["scaler"]
    lr     = pipe.named_steps["clf"]
    X_scaled = scaler.transform(X)

    # LinearExplainer（LR専用・高速）
    explainer   = shap.LinearExplainer(lr, X_scaled, feature_names=ALL_FEATURES)
    shap_values = explainer(X_scaled)

    # 多クラスの場合は放棄地クラス(1)を抽出
    if shap_values.values.ndim == 3:
        sv_aban = shap.Explanation(
            values       = shap_values.values[:, :, 1],
            base_values  = shap_values.base_values[:, 1],
            data         = shap_values.data,
            feature_names = ALL_FEATURES,
        )
    else:
        sv_aban = shap_values

    # ① Beeswarm plot（全サンプルの特徴量寄与を一枚で）
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(sv_aban, max_display=15, show=False)
    plt.title("SHAP Beeswarm - Abandoned Land Class (LR)", fontsize=13)
    plt.tight_layout()
    beeswarm_path = OUT_DIR / "shap_beeswarm_lr_abandoned.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Beeswarm → {beeswarm_path}")

    # ② Bar plot（平均|SHAP|による特徴量ランキング）
    plt.figure(figsize=(9, 6))
    shap.plots.bar(sv_aban, max_display=15, show=False)
    plt.title("SHAP Feature Importance - Abandoned Land Class (LR)", fontsize=13)
    plt.tight_layout()
    bar_path = OUT_DIR / "shap_bar_lr_abandoned.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Bar     → {bar_path}")

    # ③ RF の SHAP（TreeExplainer・GBMは多クラス非対応のためスキップ）
    for tree_name in ["RandomForest"]:
        if tree_name not in best_pipelines:
            continue
        print(f"\n=== SHAP値可視化（{tree_name}・3値分類）===")
        pipe_tree = copy.deepcopy(best_pipelines[tree_name])
        pipe_tree.fit(X, y)
        scaler_tree = pipe_tree.named_steps["scaler"]
        clf_tree    = pipe_tree.named_steps["clf"]
        X_sc        = scaler_tree.transform(X)

        # TreeExplainer（木系モデル専用・高速）
        explainer_tree = shap.TreeExplainer(clf_tree, feature_names=ALL_FEATURES)
        sv_tree        = explainer_tree(X_sc)

        # 多クラスの場合は放棄地クラス(1)を抽出
        if sv_tree.values.ndim == 3:
            sv_tree_aban = shap.Explanation(
                values        = sv_tree.values[:, :, 1],
                base_values   = sv_tree.base_values[:, 1],
                data          = sv_tree.data,
                feature_names = ALL_FEATURES,
            )
        else:
            sv_tree_aban = sv_tree

        # Beeswarm
        plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(sv_tree_aban, max_display=15, show=False)
        plt.title(f"SHAP Beeswarm - Abandoned Land Class ({tree_name})", fontsize=13)
        plt.tight_layout()
        p = OUT_DIR / f"shap_beeswarm_{tree_name.lower()}_abandoned.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Beeswarm → {p}")

        # Bar
        plt.figure(figsize=(9, 6))
        shap.plots.bar(sv_tree_aban, max_display=15, show=False)
        plt.title(f"SHAP Feature Importance - Abandoned Land Class ({tree_name})", fontsize=13)
        plt.tight_layout()
        p = OUT_DIR / f"shap_bar_{tree_name.lower()}_abandoned.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Bar     → {p}")


# =====================================================================
# メイン
# =====================================================================

def main():
    print("=" * 60)
    print("第2段階：耕作放棄地分類モデル（2値 vs 3値 比較）")
    print("=" * 60)

    # ─────────────────────────────────────────
    # A. 2値分類（畑 vs 放棄地）
    # ─────────────────────────────────────────
    print("\n" + "━" * 60)
    print("【A】2値分類: 畑(0) vs 放棄地(1)")
    print("━" * 60)

    dataset2 = prepare_dataset()
    print(f"\n  → データ数: {len(dataset2)}件")

    best_pipelines = tune_models(dataset2)

    # ② アンサンブル（GBM + RF + LightGBM soft voting）を追加
    print("\n  [Ensemble] GBM + RF + LightGBM の soft voting を構築...")
    best_pipelines["Ensemble(GBM+RF+LGBM)"] = build_ensemble_pipeline(best_pipelines)

    results2 = run_spatial_cv(dataset2, best_pipelines)
    analyze_feature_importance(dataset2, best_pipelines)

    print("\n=== 2値分類 モデル比較サマリー（全地域合算） ===")
    df2 = pd.DataFrame(results2)
    df2_disp = df2.copy()
    for col in ["acc", "f1", "prec", "rec", "iou"]:
        df2_disp[col] = (df2_disp[col] * 100).map("{:.2f}%".format)
    df2_disp.columns = ["モデル", "Accuracy", "F1", "Precision", "Recall", "IoU"]
    print(df2_disp.to_string(index=False))

    # ─────────────────────────────────────────
    # B. 3値分類（畑 vs 放棄地 vs 水田）
    # ─────────────────────────────────────────
    print("\n" + "━" * 60)
    print("【B】3値分類: 畑(0) vs 放棄地(1) vs 水田(2)")
    print("━" * 60)

    dataset3 = prepare_dataset_3class()
    print(f"\n  → データ数: {len(dataset3)}件")

    # 3値分類はチューニング済みパイプラインをそのまま流用
    # （パラメータ空間は同一なので再チューニングは行わない）
    results3 = run_spatial_cv_3class(dataset3, best_pipelines)

    # ─────────────────────────────────────────
    # C. 2値 vs 3値 比較サマリー
    # ─────────────────────────────────────────
    print("\n" + "━" * 60)
    print("【C】2値 vs 3値 放棄地クラス性能比較（全地域合算）")
    print("━" * 60)
    print(f"\n  {'モデル':<20} {'2値-F1':>8} {'2値-IoU':>9} {'3値-F1':>8} {'3値-IoU':>9} {'ΔF1':>7}")
    print("  " + "─" * 65)

    res2_map = {r["model"]: r for r in results2}
    for r3 in results3:
        r2    = res2_map.get(r3["model"])
        if r2 is None:
            continue
        delta = (r3["f1_aban"] - r2["f1"]) * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {r3['model']:<20} {r2['f1']*100:>7.2f}% {r2['iou']*100:>8.2f}%"
              f" {r3['f1_aban']*100:>7.2f}% {r3['iou_aban']*100:>8.2f}%"
              f" {sign}{delta:>5.2f}%")

    # ─────────────────────────────────────────
    # D. GBM 確信度スコア精度検証
    # ─────────────────────────────────────────
    print("\n" + "━" * 60)
    print("【D】GBM 確信度スコア精度検証（高/中/低）")
    print("━" * 60)
    gbm_pipe = best_pipelines.get("GradientBoosting")
    if gbm_pipe is not None:
        analyze_confidence_precision(dataset2, gbm_pipe)
    else:
        print("  GradientBoosting パイプラインが見つかりません")

    # ─────────────────────────────────────────
    # E. 閾値チューニング（行政用 vs 現地調査用）
    # ─────────────────────────────────────────
    print("\n" + "━" * 60)
    print("【E】分類閾値チューニング（0.30〜0.70）")
    print("━" * 60)
    if gbm_pipe is not None:
        analyze_threshold(dataset2, gbm_pipe)
    else:
        print("  GradientBoosting パイプラインが見つかりません")

    # ─────────────────────────────────────────
    # F. 特徴量アブレーション比較
    # ─────────────────────────────────────────
    print("\n" + "━" * 60)
    print("【F】特徴量アブレーション（全19 vs SHAP上位8）")
    print("━" * 60)
    run_ablation_comparison(dataset3, best_pipelines)

    # ─────────────────────────────────────────
    # G. SHAP値可視化（LR・3値分類）
    # ─────────────────────────────────────────
    print("\n" + "━" * 60)
    print("【G】SHAP値可視化（LogisticRegression・放棄地クラス）")
    print("━" * 60)
    analyze_shap(dataset3, best_pipelines)

    print("\n" + "=" * 60)
    print("完了")

    return best_pipelines, dataset2, dataset3


if __name__ == "__main__":
    main()
