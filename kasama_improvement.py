"""
笠間市 水田スクリーニング精度改善実験
ベースライン IoU 70.1% からの改善を複数手法で検証
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.family"] = "Hiragino Sans"
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_PATH = "/Users/nagatomotaito/Documents/rice_research/kasama_features.csv"
BASE_FEATURES = ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"]
LABEL_COL = "mode"
RICE_CLASS = 3
RANDOM_STATE = 42


# =========================================================
# データ読み込み・前処理
# =========================================================
df = pd.read_csv(DATA_PATH).dropna(subset=BASE_FEATURES + [LABEL_COL])
y_true = (df[LABEL_COL].values == RICE_CLASS).astype(int)
print(f"サンプル数: {len(df):,}  水田比率: {y_true.mean():.3f}")


# =========================================================
# 評価関数
# =========================================================
def iou(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    union = tp + fp + fn
    return float(tp / union) if union > 0 else 0.0

def evaluate(y_true, labels, k, method="multi_threshold"):
    """クラスタラベルから最適な水田判定を行い評価指標を返す"""
    rice_ratio = np.array([
        y_true[labels == c].mean() if (labels == c).any() else 0.0
        for c in range(k)
    ])
    if method == "best_single":
        # 最も水田比率が高い1クラスタのみ
        best = int(np.argmax(rice_ratio))
        y_pred = (labels == best).astype(int)
    else:
        # 水田比率 > 0.5 のクラスタを全て水田とする
        rice_clusters = set(np.where(rice_ratio > 0.5)[0])
        if not rice_clusters:
            rice_clusters = {int(np.argmax(rice_ratio))}
        y_pred = np.isin(labels, list(rice_clusters)).astype(int)

    acc = accuracy_score(y_true, y_pred)
    i = iou(y_true, y_pred)
    return acc, i, y_pred


# =========================================================
# 特徴量セットの定義
# =========================================================
df["dNDVI"] = df["NDVI_grow"] - df["NDVI_flood"]      # 生育期-湛水期 NDVI差
df["dVH"]   = df["VH_min"] - df["VH_winter"]           # 湛水時-冬期 VH差
df["lat_n"] = df["point_lat"]                           # 緯度（地形プロキシ）
df["lng_n"] = df["point_lng"]                           # 経度

FEATURE_SETS = {
    "ベースライン (4特徴量)":
        ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"],
    "+dNDVI (NDVI振幅)":
        ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood", "dNDVI"],
    "+dVH (湛水シグナル差)":
        ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood", "dVH"],
    "+dNDVI+dVH":
        ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood", "dNDVI", "dVH"],
    "+空間座標":
        ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood", "lat_n", "lng_n"],
    "+dNDVI+dVH+空間":
        ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood", "dNDVI", "dVH", "lat_n", "lng_n"],
}


# =========================================================
# 実験1: 特徴量 × K値 × アルゴリズム の組み合わせ
# =========================================================
results = []

for feat_name, features in FEATURE_SETS.items():
    X = df[features].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # K-Means (k=3〜8)
    for k in range(3, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_s)
        for method in ["best_single", "multi_threshold"]:
            acc, i, _ = evaluate(y_true, labels, k, method)
            results.append({
                "特徴量セット": feat_name,
                "アルゴリズム": f"K-Means(k={k})",
                "判定方式": method,
                "Accuracy": round(acc, 4),
                "IoU": round(i, 4),
            })

    # GMM (k=3〜6)
    for k in range(3, 7):
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=3)
        labels = gmm.fit_predict(X_s)
        for method in ["best_single", "multi_threshold"]:
            acc, i, _ = evaluate(y_true, labels, k, method)
            results.append({
                "特徴量セット": feat_name,
                "アルゴリズム": f"GMM(k={k})",
                "判定方式": method,
                "Accuracy": round(acc, 4),
                "IoU": round(i, 4),
            })

    # 階層クラスタリング (k=4,5)
    for k in [4, 5]:
        ag = AgglomerativeClustering(n_clusters=k)
        labels = ag.fit_predict(X_s)
        for method in ["best_single", "multi_threshold"]:
            acc, i, _ = evaluate(y_true, labels, k, method)
            results.append({
                "特徴量セット": feat_name,
                "アルゴリズム": f"Agglomerative(k={k})",
                "判定方式": method,
                "Accuracy": round(acc, 4),
                "IoU": round(i, 4),
            })

df_res = pd.DataFrame(results)


# =========================================================
# 結果の整理・表示
# =========================================================
BASELINE_IOU = 0.701

print("\n" + "="*65)
print("【IoU上位10件】")
print("="*65)
top10 = df_res.sort_values("IoU", ascending=False).head(10)
print(top10[["特徴量セット","アルゴリズム","判定方式","Accuracy","IoU"]].to_string(index=False))

print("\n" + "="*65)
print("【特徴量セット別 最良IoU】")
print("="*65)
best_per_feat = df_res.groupby("特徴量セット")["IoU"].max().sort_values(ascending=False)
for feat, val in best_per_feat.items():
    delta = val - BASELINE_IOU
    mark = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "→")
    print(f"  {mark} {val:.4f} ({delta:+.4f})  {feat}")


# =========================================================
# ベスト手法の詳細評価
# =========================================================
best_row = df_res.loc[df_res["IoU"].idxmax()]
print(f"\n{'='*65}")
print(f"【ベスト手法】")
print(f"  特徴量セット : {best_row['特徴量セット']}")
print(f"  アルゴリズム : {best_row['アルゴリズム']}")
print(f"  判定方式     : {best_row['判定方式']}")
print(f"  Accuracy     : {best_row['Accuracy']:.4f}")
print(f"  IoU          : {best_row['IoU']:.4f}  (ベースライン比 {best_row['IoU']-BASELINE_IOU:+.4f})")


# =========================================================
# 可視化: 特徴量セット別 IoU 比較棒グラフ
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("笠間市 精度改善実験", fontsize=14)

# (1) 特徴量セット別最良IoU
feat_best = df_res.groupby("特徴量セット")["IoU"].max().sort_values()
colors = ["#4C72B0" if v > BASELINE_IOU else "#DD8452" for v in feat_best.values]
axes[0].barh(feat_best.index, feat_best.values, color=colors)
axes[0].axvline(BASELINE_IOU, color="red", linestyle="--", label=f"ベースライン {BASELINE_IOU}")
axes[0].set_xlabel("IoU")
axes[0].set_title("特徴量セット別 最良IoU")
axes[0].legend()
axes[0].set_xlim(0.6, min(1.0, feat_best.max()+0.05))

# (2) K-Means k値別 IoU (各特徴量セット)
ax2 = axes[1]
k_vals = list(range(3, 9))
for feat_name in FEATURE_SETS.keys():
    iou_per_k = []
    for k in k_vals:
        sub = df_res[
            (df_res["特徴量セット"] == feat_name) &
            (df_res["アルゴリズム"] == f"K-Means(k={k})")
        ]["IoU"].max()
        iou_per_k.append(sub)
    lw = 2.5 if "dNDVI" in feat_name or "dVH" in feat_name else 1
    ax2.plot(k_vals, iou_per_k, marker="o", linewidth=lw, label=feat_name)

ax2.axhline(BASELINE_IOU, color="red", linestyle="--", label=f"ベースライン")
ax2.set_xlabel("k")
ax2.set_ylabel("IoU")
ax2.set_title("K-Means: k値 vs IoU")
ax2.legend(fontsize=7, loc="lower right")
ax2.set_xticks(k_vals)

plt.tight_layout()
out = "/Users/nagatomotaito/Documents/rice_research/kasama_improvement.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n図を保存: {out}")
plt.show()
