"""
3地域 精度改善実験（笠間市・稲敷市・つくばみらい市）
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.family"] = "Hiragino Sans"
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

BASE_DIR = "/Users/nagatomotaito/Documents/rice_research/"
REGIONS = {
    "笠間市":        {"file": "kasama_features.csv",      "baseline_iou": 0.701},
    "稲敷市":        {"file": "inashiki_features.csv",    "baseline_iou": 0.893},
    "つくばみらい市": {"file": "tsukubamirai_features.csv","baseline_iou": 0.801},
}
BASE_FEATURES = ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"]
LABEL_COL = "mode"
RICE_CLASS = 3
RANDOM_STATE = 42


def iou(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    union = tp + fp + fn
    return float(tp / union) if union > 0 else 0.0


def evaluate(y_true, labels, k):
    rice_ratio = np.array([
        y_true[labels == c].mean() if (labels == c).any() else 0.0
        for c in range(k)
    ])
    # best_single
    best = int(np.argmax(rice_ratio))
    y_pred_s = (labels == best).astype(int)
    iou_s = iou(y_true, y_pred_s)
    acc_s = accuracy_score(y_true, y_pred_s)
    # multi_threshold
    rice_clusters = set(np.where(rice_ratio > 0.5)[0])
    if not rice_clusters:
        rice_clusters = {best}
    y_pred_m = np.isin(labels, list(rice_clusters)).astype(int)
    iou_m = iou(y_true, y_pred_m)
    acc_m = accuracy_score(y_true, y_pred_m)
    return max(iou_s, iou_m), max(acc_s, acc_m)


def run_experiments(df, y_true):
    df = df.copy()
    df["dNDVI"] = df["NDVI_grow"] - df["NDVI_flood"]
    df["dVH"]   = df["VH_min"] - df["VH_winter"]

    FEATURE_SETS = {
        "ベースライン":      BASE_FEATURES,
        "+dNDVI":           BASE_FEATURES + ["dNDVI"],
        "+dVH":             BASE_FEATURES + ["dVH"],
        "+dNDVI+dVH":       BASE_FEATURES + ["dNDVI", "dVH"],
        "+空間座標":         BASE_FEATURES + ["point_lat", "point_lng"],
        "+dNDVI+dVH+空間":  BASE_FEATURES + ["dNDVI", "dVH", "point_lat", "point_lng"],
    }

    results = []
    for feat_name, features in FEATURE_SETS.items():
        X_s = StandardScaler().fit_transform(df[features].values)

        for k in range(3, 9):
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            i, a = evaluate(y_true, km.fit_predict(X_s), k)
            results.append({"特徴量": feat_name, "アルゴリズム": f"K-Means(k={k})", "IoU": i, "Accuracy": a})

        for k in range(3, 7):
            gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=3)
            i, a = evaluate(y_true, gmm.fit_predict(X_s), k)
            results.append({"特徴量": feat_name, "アルゴリズム": f"GMM(k={k})", "IoU": i, "Accuracy": a})

        # AgglomerativeClustering は大規模データに非適 → 除外

    return pd.DataFrame(results)


# =========================================================
# 全地域実験
# =========================================================
all_results = {}
region_summary = []

for region, info in REGIONS.items():
    print(f"\n{'='*50}")
    print(f"  {region} を処理中...")
    df = pd.read_csv(BASE_DIR + info["file"]).dropna(subset=BASE_FEATURES + [LABEL_COL])
    y_true = (df[LABEL_COL].values == RICE_CLASS).astype(int)
    print(f"  サンプル数: {len(df):,}  水田比率: {y_true.mean():.3f}")

    res = run_experiments(df, y_true)
    all_results[region] = res

    best = res.loc[res["IoU"].idxmax()]
    baseline = info["baseline_iou"]
    delta = best["IoU"] - baseline
    region_summary.append({
        "地域": region,
        "ベースライン IoU": baseline,
        "最良 IoU": round(best["IoU"], 4),
        "改善幅": round(delta, 4),
        "最良特徴量": best["特徴量"],
        "最良アルゴリズム": best["アルゴリズム"],
    })
    print(f"  ベースライン: {baseline:.4f}  →  最良: {best['IoU']:.4f}  ({delta:+.4f})")
    print(f"  最良手法: {best['特徴量']} × {best['アルゴリズム']}")


# =========================================================
# サマリー表示
# =========================================================
df_summary = pd.DataFrame(region_summary)
print("\n" + "="*60)
print("【3地域 精度改善サマリー】")
print("="*60)
print(df_summary.to_string(index=False))


# =========================================================
# 可視化
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("3地域 精度改善実験：特徴量セット別 最良IoU比較", fontsize=14)

for ax, (region, info) in zip(axes, REGIONS.items()):
    res = all_results[region]
    feat_best = res.groupby("特徴量")["IoU"].max().sort_values()
    baseline = info["baseline_iou"]
    colors = ["#4C72B0" if v > baseline else "#DD8452" for v in feat_best.values]
    ax.barh(feat_best.index, feat_best.values, color=colors)
    ax.axvline(baseline, color="red", linestyle="--", linewidth=1.5, label=f"ベースライン {baseline:.3f}")
    ax.set_title(region, fontsize=13)
    ax.set_xlabel("IoU")
    ax.legend(fontsize=9)
    lo = max(0, min(feat_best.values) - 0.02)
    hi = min(1.0, max(feat_best.values) + 0.03)
    ax.set_xlim(lo, hi)

plt.tight_layout()
out = BASE_DIR + "all_regions_improvement.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n図を保存: {out}")
plt.show()
