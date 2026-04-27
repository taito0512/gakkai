"""
2段階農地分類パイプライン
=====================================
第1段階: K-Means（教師なし）→ 水田 / 非水田 を判定
第2段階: 教師あり分類器     → 非水田に対して分類
  mode='2class': 畑(0) / 放棄地(1)
  mode='3class': 畑(0) / 放棄地(1) / 水田(2) ← 第1段階取り逃がし補正
    ・水田訓練データ: ラベルCSV水田っぽい(30件) + K-Means高確信度疑似ラベル(300件)
    ・水田と予測されたら第1段階の非水田判定を上書きして水田に補正
    ・補正精度(空間CV): Precision 97.7% / Recall 91.2%（GradientBoosting）

使い方:
    python pipeline.py                         # 全地域・3class（推奨）
    python pipeline.py --region 稲敷市
    python pipeline.py --mode 2class           # 2値分類
    python pipeline.py --mode both             # 2値と3値を比較
"""

import argparse
import copy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# --- 設定 ---
LABELS_CSV   = "labels/combined_labels.csv"
FEATURE_CSVS = {
    "つくばみらい市": "data/tsukubamirai_features.csv",
    "稲敷市":         "data/inashiki_features.csv",
    "笠間市":         "data/kasama_features.csv",
}

# --- 第1段階設定（K-Means） ---
STAGE1_FEATURES  = ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"]
STAGE1_K         = 4
RICE_LABEL_COL   = "mode"
RICE_CLASS_VAL   = 3

# --- 第2段階設定（教師あり分類） ---
BASE_FEATURES    = ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"]
STAGE2_FEATURES  = BASE_FEATURES + ["dNDVI", "dVH", "area_m2"]
COORD_THRESHOLD  = 0.001  # 度（約100m）
RANDOM_STATE     = 42


# =====================================================================
# ユーティリティ
# =====================================================================

def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    union = tp + fp + fn
    return float(tp / union) if union > 0 else 0.0


def classify_memo(memo) -> str:
    if pd.isna(memo):
        return "不明"
    memo = str(memo)
    if any(k in memo for k in ["水田", "緑色管理圃場", "湛水", "畦", "冠水"]):
        return "水田っぽい"
    if any(k in memo for k in [
        "褐色耕起", "畝列", "列作物", "トラクター", "施設園芸", "畑",
        "マルチ", "ハウス", "均一な列", "管理明確"
    ]):
        return "畑っぽい"
    return "不明"


# =====================================================================
# 第1段階: K-Means 水田スクリーニング
# =====================================================================

class Stage1KMeans:
    """
    K-Means による水田スクリーニング。
    VH_min が最小のクラスタを水田と判定（教師データ不要）。
    """

    def __init__(self, k: int = STAGE1_K, random_state: int = RANDOM_STATE):
        self.k            = k
        self.random_state = random_state
        self.scaler_      = None
        self.kmeans_      = None

    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        特徴量DataFrameをfit＆predict。
        Returns: 1次元配列（1=水田, 0=非水田）
        """
        df = df.copy()
        df["dNDVI"] = df["NDVI_grow"] - df["NDVI_flood"]
        df["dVH"]   = df["VH_min"]    - df["VH_winter"]

        X = df[STAGE1_FEATURES].values  # K-Meansは基本4特徴量のみ

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.kmeans_ = KMeans(
            n_clusters=self.k, random_state=self.random_state, n_init=10)
        cluster_labels = self.kmeans_.fit_predict(X_scaled)

        # VH_min（index=0）の平均が最小のクラスタ → 水田
        vh_min_mean = np.array([
            X[cluster_labels == c, 0].mean() if (cluster_labels == c).any() else np.inf
            for c in range(self.k)
        ])
        rice_cluster = int(np.argmin(vh_min_mean))
        return (cluster_labels == rice_cluster).astype(int)


# =====================================================================
# 第2段階: 教師あり畑/放棄地分類
# =====================================================================

class Stage2AbandonedClassifier:
    """
    教師あり分類（畑=0 / 放棄地=1 / 水田=2）。
    mode='2class': 畑 vs 放棄地（2値）
    mode='3class': 畑 vs 放棄地 vs 水田（3値・第1段階の取り逃がし補正用）
    combined_labels.csv から訓練データを取得し座標マッチングで特徴量と結合。
    """

    def __init__(self, mode: str = "3class"):
        assert mode in ("2class", "3class"), "mode は '2class' か '3class'"
        self.mode      = mode
        self.pipeline_ = None  # 訓練済みパイプライン（StandardScaler + 分類器）

    def _build_pipeline(self) -> Pipeline:
        """
        空間CV検証済みのベストモデルを使用。
        2class / 3class ともに GradientBoosting が最良（F1: 84.92% / 82.94%）。
        """
        clf = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    def _get_pseudo_rice_samples(self, n_per_region: int = 100) -> pd.DataFrame:
        """
        K-Means高確信度水田を疑似ラベル（class=2）として取得する。
        各地域でK-Meansを実行し、水田クラスタ中心から距離が下位33%の
        サンプルを最大 n_per_region 件ランダムサンプリングして返す。
        3値分類の水田クラス補充用（30件 → +300件）。
        """
        rng        = np.random.default_rng(RANDOM_STATE)
        all_rows   = []

        for region, feat_path in FEATURE_CSVS.items():
            feat_df = pd.read_csv(feat_path, low_memory=False)
            feat_df["dNDVI"] = feat_df["NDVI_grow"] - feat_df["NDVI_flood"]
            feat_df["dVH"]   = feat_df["VH_min"]    - feat_df["VH_winter"]
            if "area_m2" not in feat_df.columns:
                feat_df["area_m2"] = np.nan

            # 基本特徴量でNaN除去
            clean = feat_df.dropna(
                subset=BASE_FEATURES + ["dNDVI", "dVH"]
            ).reset_index(drop=True)
            clean["area_m2"] = pd.to_numeric(
                clean["area_m2"], errors="coerce").fillna(1000.0)

            # K-Means実行（4クラスタ・基本4特徴量）
            X4       = clean[BASE_FEATURES].values
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X4)
            km       = KMeans(n_clusters=STAGE1_K, random_state=RANDOM_STATE, n_init=10)
            km.fit(X_scaled)
            c_labels = km.labels_

            # VH_min最小クラスタ → 水田
            vh_means     = np.array([
                X4[c_labels == c, 0].mean() if (c_labels == c).any() else np.inf
                for c in range(STAGE1_K)
            ])
            rice_cluster = int(np.argmin(vh_means))
            rice_idx     = np.where(c_labels == rice_cluster)[0]

            # クラスタ中心距離 下位33%を高確信度とする
            center    = km.cluster_centers_[rice_cluster]
            dists     = np.linalg.norm(X_scaled[rice_idx] - center, axis=1)
            threshold = np.percentile(dists, 33)
            hc_idx    = rice_idx[dists <= threshold]

            sampled = (rng.choice(hc_idx, size=n_per_region, replace=False)
                       if len(hc_idx) > n_per_region else hc_idx)

            rows          = clean.iloc[sampled].copy()
            rows["地域"]   = region
            rows["クラス"] = 2
            all_rows.append(rows)

        return pd.concat(all_rows, ignore_index=True)

    def _load_training_data(self) -> pd.DataFrame:
        """combined_labels.csv から訓練データを構築する。
        3class モードでは K-Means 疑似ラベル水田（300件）を追加する。"""
        labels = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
        labels.rename(columns={"面積m2": "area_m2"}, inplace=True)

        abandoned = labels[labels["ラベル"] == "放棄地"].copy()
        abandoned["クラス"] = 1

        working = labels[labels["ラベル"] == "耕作中"].copy()
        working["memo_type"] = working["メモ"].apply(classify_memo)
        hatake = working[working["memo_type"] == "畑っぽい"].copy()
        hatake["クラス"] = 0

        parts = [abandoned, hatake]

        if self.mode == "3class":
            # ラベルCSVの水田っぽい（30件）を追加
            suiden = working[working["memo_type"] == "水田っぽい"].copy()
            suiden["クラス"] = 2
            parts.append(suiden)

        label_combined = pd.concat(parts, ignore_index=True)

        # 地域ごとに特徴量マッチング
        all_rows = []
        for region, feat_path in FEATURE_CSVS.items():
            subset = label_combined[label_combined["地域"] == region].copy()
            if len(subset) == 0:
                continue
            feat_df = pd.read_csv(feat_path, low_memory=False)
            feat_df["dNDVI"] = feat_df["NDVI_grow"] - feat_df["NDVI_flood"]
            feat_df["dVH"]   = feat_df["VH_min"]    - feat_df["VH_winter"]

            tree = cKDTree(feat_df[["point_lat", "point_lng"]].values)
            dists, idxs = tree.query(subset[["lat", "lon"]].values, k=1)

            for col in BASE_FEATURES + ["dNDVI", "dVH"]:
                subset = subset.copy()
                subset[col] = np.where(
                    dists < COORD_THRESHOLD,
                    feat_df.iloc[idxs][col].values,
                    np.nan
                )
            all_rows.append(subset)

        train_df = pd.concat(all_rows, ignore_index=True)
        train_df["area_m2"] = pd.to_numeric(train_df["area_m2"], errors="coerce")
        train_df = train_df.dropna(subset=STAGE2_FEATURES)

        # 3class モード: K-Means 疑似ラベル水田（地域ごと100件 × 3地域）を追加
        if self.mode == "3class":
            pseudo = self._get_pseudo_rice_samples(n_per_region=100)
            pseudo = pseudo.dropna(subset=STAGE2_FEATURES)
            train_df = pd.concat([train_df, pseudo], ignore_index=True)

        return train_df

    def fit(self):
        """全教師データでモデルを訓練する"""
        train_df = self._load_training_data()
        X = train_df[STAGE2_FEATURES]  # DataFrameのまま渡す（LGBMのfeature名警告を防ぐ）
        y = train_df["クラス"].values
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)
        label_str = f"畑{(y==0).sum()} / 放棄地{(y==1).sum()}"
        if self.mode == "3class":
            label_str += f" / 水田{(y==2).sum()}（ラベル+疑似）"
        print(f"  第2段階モデル訓練完了 ({self.mode}, n={len(train_df)}: {label_str})")
        return self

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        """
        特徴量DataFrame（全筆）に対して予測。
        Returns: 1次元配列（0=畑, 1=放棄地）
        """
        assert self.pipeline_ is not None, "先に fit() を呼んでください"
        df = feat_df.copy()
        df["dNDVI"] = df["NDVI_grow"] - df["NDVI_flood"]
        df["dVH"]   = df["VH_min"]    - df["VH_winter"]
        df["area_m2"] = pd.to_numeric(df.get("area_m2", np.nan), errors="coerce").fillna(
            df["area_m2"].median() if "area_m2" in df.columns else 1000.0
        )
        X = df[STAGE2_FEATURES]  # DataFrameのまま渡す
        return self.pipeline_.predict(X)

    def predict_proba(self, feat_df: pd.DataFrame) -> np.ndarray:
        """
        放棄地クラス(1)の確率スコアを返す（確信度として利用可能）。
        LGBMClassifier/GBMはpredict_probaをサポート。
        """
        assert self.pipeline_ is not None
        df = feat_df.copy()
        df["dNDVI"] = df["NDVI_grow"] - df["NDVI_flood"]
        df["dVH"]   = df["VH_min"]    - df["VH_winter"]
        df["area_m2"] = pd.to_numeric(df.get("area_m2", np.nan), errors="coerce").fillna(
            df["area_m2"].median() if "area_m2" in df.columns else 1000.0
        )
        X = df[STAGE2_FEATURES]  # DataFrameのまま渡す
        return self.pipeline_.predict_proba(X)[:, 1]


# =====================================================================
# 統合パイプライン
# =====================================================================

class TwoStagePipeline:
    """
    第1段階（K-Means水田判定）+ 第2段階（畑/放棄地[/水田]分類）の統合パイプライン。

    mode='2class': 第2段階は畑/放棄地の2値分類
    mode='3class': 第2段階は畑/放棄地/水田の3値分類
                   → 水田と予測されたものは第1段階の結果を上書きして水田に補正
    """

    def __init__(self, mode: str = "3class"):
        self.mode   = mode
        self.stage1 = Stage1KMeans()
        self.stage2 = Stage2AbandonedClassifier(mode=mode)

    def fit_stage2(self):
        """第2段階モデルを訓練（第1段階はfit不要・教師なし）"""
        print(f"第2段階モデルを訓練中... (mode={self.mode})")
        self.stage2.fit()
        return self

    def predict(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量DataFrame（1地域分）に対して2段階予測を実行。

        Returns: 元の DataFrame に以下の列を追加したもの
          - stage1_rice:     1=水田, 0=非水田
          - stage2_pred:     0=畑, 1=放棄地, 2=水田（3値モードのみ）
          - stage2_prob:     放棄地確率
          - final_label:     水田 / 畑 / 放棄地
          - rice_corrected:  True = 第2段階が水田と判定して上書き（3値モードのみ）
        """
        df = feat_df.copy()
        df["dNDVI"] = df["NDVI_grow"] - df["NDVI_flood"]
        df["dVH"]   = df["VH_min"]    - df["VH_winter"]

        # --- 第1段階 ---
        df["stage1_rice"]    = self.stage1.fit_predict(df)
        df["rice_corrected"] = False

        # --- 第2段階（非水田のみ） ---
        non_rice_mask     = df["stage1_rice"] == 0
        df["stage2_pred"] = np.nan
        df["stage2_prob"] = np.nan

        if non_rice_mask.sum() > 0:
            non_rice_df = df[non_rice_mask].copy()
            if "area_m2" not in non_rice_df.columns:
                non_rice_df["area_m2"] = 1000.0

            preds = self.stage2.predict(non_rice_df)
            probs = self.stage2.predict_proba(non_rice_df)
            df.loc[non_rice_mask, "stage2_pred"] = preds
            df.loc[non_rice_mask, "stage2_prob"] = probs

            # 3値モード: 水田(2)と予測されたものを水田に上書き補正
            if self.mode == "3class":
                corrected_mask = non_rice_mask & (df["stage2_pred"] == 2)
                df.loc[corrected_mask, "stage1_rice"]    = 1
                df.loc[corrected_mask, "rice_corrected"] = True

        # --- 最終ラベル ---
        def assign_label(row):
            if row["stage1_rice"] == 1:
                return "水田"
            if row["stage2_pred"] == 1:
                return "放棄地"
            if row["stage2_pred"] == 0:
                return "畑"
            return "非農地"

        df["final_label"] = df.apply(assign_label, axis=1)

        # --- ③ 確信度スコア（放棄地予測筆の確率分布でパーセンタイル3段階）---
        # K-Meansの確信度と同じ考え方:
        # 「放棄地と予測した筆」の中で確率が高い順に高/中/低に分ける
        aban_mask = df["final_label"] == "放棄地"
        if aban_mask.sum() > 0:
            aban_probs = df.loc[aban_mask, "stage2_prob"]
            p33 = aban_probs.quantile(0.33)
            p66 = aban_probs.quantile(0.66)
        else:
            p33, p66 = 0.5, 0.7

        def assign_conf(row):
            if row["final_label"] != "放棄地":
                return ""
            p = row["stage2_prob"]
            if pd.isna(p):
                return ""
            if p >= p66:
                return "高"
            if p >= p33:
                return "中"
            return "低"

        df["abandoned_conf"] = df.apply(assign_conf, axis=1)
        return df

    def evaluate(self, df: pd.DataFrame):
        """
        第1段階・第2段階・統合の精度を出力する（正解ラベルが必要）。
        第1段階: mode列（JAXAラベル）で水田=1/非水田=0
        第2段階: combined_labels.csv と座標マッチング（概算評価）
        """
        print("\n--- 第1段階精度（水田 vs 非水田）---")
        if RICE_LABEL_COL not in df.columns:
            print("  mode列なし → 評価スキップ")
            return

        # accuracy_check.py と同様に mode 欠損行を除外してから評価
        valid = df[df[RICE_LABEL_COL].notna()].copy()
        print(f"  評価対象: {len(valid):,}筆（mode欠損除外後）")

        y_true_rice = (valid[RICE_LABEL_COL] == RICE_CLASS_VAL).astype(int).values
        y_pred_rice = valid["stage1_rice"].values

        acc = accuracy_score(y_true_rice, y_pred_rice)
        iou = compute_iou(y_true_rice, y_pred_rice)
        cm  = confusion_matrix(y_true_rice, y_pred_rice)
        tn, fp, fn, tp = cm.ravel()
        print(f"  Accuracy: {acc*100:.2f}%  IoU: {iou*100:.2f}%")
        print(f"  TP:{tp:,} FP:{fp:,} FN:{fn:,} TN:{tn:,}")


# =====================================================================
# CLI エントリポイント
# =====================================================================

def run_region(region: str, pipeline: TwoStagePipeline):
    """1地域の予測を実行して結果を表示"""
    feat_df    = pd.read_csv(FEATURE_CSVS[region], low_memory=False)
    result_df  = pipeline.predict(feat_df)
    label_counts = result_df["final_label"].value_counts()
    total = len(result_df)

    print(f"  【{region}】")
    for label in ["水田", "畑", "放棄地"]:
        count = label_counts.get(label, 0)
        print(f"    {label:<5}: {count:>7,}筆  ({count/total*100:.1f}%)")

    if pipeline.mode == "3class":
        n_corrected = result_df["rice_corrected"].sum()
        print(f"    ↑ 水田補正: {n_corrected:,}筆を第2段階で水田に修正")

    pipeline.evaluate(result_df)

    # ③ 確信度スコア別の放棄地内訳
    print("\n--- 確信度スコア別 放棄地内訳 ---")
    abandoned_df = result_df[result_df["final_label"] == "放棄地"]
    total_aban   = len(abandoned_df)
    for level in ["高", "中", "低"]:
        n   = (abandoned_df["abandoned_conf"] == level).sum()
        pct = n / total_aban * 100 if total_aban > 0 else 0
        mark = {"高": "🟢", "中": "🟡", "低": "🔴"}.get(level, "")
        print(f"  {mark} {level}確信度: {n:>6,}筆  ({pct:.1f}%)")
    print(f"  合計放棄地: {total_aban:>6,}筆")
    print("  ※ 高確信度 = 放棄地確率 上位33%（現地確認優先度: 高）")

    out_path = f"results_{region}_{pipeline.mode}.csv"
    cols_to_save = [c for c in result_df.columns
                    if c not in ["history", ".geo", "last_polyg", "prev_last_"]]
    result_df[cols_to_save].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"    → 保存: {out_path}\n")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="2段階農地分類パイプライン")
    parser.add_argument("--region", default=None,
                        help="地域名（省略時は全地域）: つくばみらい市 / 稲敷市 / 笠間市")
    parser.add_argument("--mode", default="both",
                        choices=["2class", "3class", "both"],
                        help="2class: 畑/放棄地, 3class: 畑/放棄地/水田補正, both: 両方比較")
    args = parser.parse_args()

    regions = [args.region] if args.region else list(FEATURE_CSVS.keys())
    if args.region and args.region not in FEATURE_CSVS:
        print(f"エラー: 使用可能地域 = {list(FEATURE_CSVS.keys())}")
        return

    modes = ["2class", "3class"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print("=" * 55)
        print(f"2段階農地分類パイプライン  mode={mode}")
        print("  第1段階: K-Means（水田スクリーニング）")
        if mode == "2class":
            print("  第2段階: 畑 vs 放棄地（2値）")
        else:
            print("  第2段階: 畑 vs 放棄地 vs 水田（3値・上書き補正あり）")
        print("=" * 55)

        pipe = TwoStagePipeline(mode=mode)
        pipe.fit_stage2()

        for region in regions:
            run_region(region, pipe)

    print("完了")


if __name__ == "__main__":
    main()
