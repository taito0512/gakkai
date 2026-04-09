"""
水田スクリーニング判別アプリ
Sentinel-1/2特徴量を用いた教師なしクラスタリングによる水田判別
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.family"] = "Hiragino Sans"
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import folium
from streamlit_folium import st_folium

# =========================================================
# 設定
# =========================================================
from pathlib import Path
BASE_DIR = Path(__file__).parent  # ローカルでもStreamlit Cloudでも動く相対パス

REGIONS = {
    "つくばみらい市": str(BASE_DIR / "tsukubamirai_features.csv"),
    "稲敷市":        str(BASE_DIR / "inashiki_features.csv"),
    "笠間市":        str(BASE_DIR / "kasama_features.csv"),
}

FEATURES = ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"]
LABEL_COL = "mode"
RICE_CLASS = 3
RANDOM_STATE = 42

# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="水田スクリーニング判別アプリ",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 水田スクリーニング判別アプリ")
st.caption("Sentinel-1/2衛星データを用いた教師なしクラスタリングによる水田判別")

# =========================================================
# サイドバー
# =========================================================
st.sidebar.header("⚙️ 設定")

region_option = st.sidebar.selectbox(
    "地域選択",
    ["つくばみらい市", "稲敷市", "笠間市", "全地域"]
)

k = st.sidebar.slider("クラスタ数 k", min_value=3, max_value=10, value=5)

method = st.sidebar.radio("クラスタリング手法", ["K-Means", "GMM"])

run_btn = st.sidebar.button("▶ 実行", type="primary", use_container_width=True)

# =========================================================
# データ読み込み
# =========================================================
@st.cache_data
def load_data(region_option):
    if region_option == "全地域":
        dfs = []
        for name, path in REGIONS.items():
            df = pd.read_csv(path)
            df["地域"] = name
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(REGIONS[region_option])
        df["地域"] = region_option
        return df

# =========================================================
# クラスタリング・評価
# =========================================================
def run_clustering(df, k, method):
    df = df.dropna(subset=FEATURES + [LABEL_COL]).copy()
    X = StandardScaler().fit_transform(df[FEATURES].values)
    y_true = (df[LABEL_COL].values == RICE_CLASS).astype(int)

    if method == "K-Means":
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    else:
        model = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=3)

    labels = model.fit_predict(X)

    # 水田比率が最も高いクラスタを水田と判定
    rice_ratio = np.array([
        y_true[labels == c].mean() if (labels == c).any() else 0.0
        for c in range(k)
    ])
    best_cluster = int(np.argmax(rice_ratio))
    y_pred = (labels == best_cluster).astype(int)

    df["cluster"] = labels
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    # 分類タイプ（TP/FP/FN/TN）
    def classify(row):
        if row["y_true"] == 1 and row["y_pred"] == 1:
            return "TP"
        elif row["y_true"] == 0 and row["y_pred"] == 1:
            return "FP"
        elif row["y_true"] == 1 and row["y_pred"] == 0:
            return "FN"
        else:
            return "TN"
    df["分類"] = df.apply(classify, axis=1)

    return df, y_true, y_pred, labels

def calc_iou(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    union = tp + fp + fn
    return float(tp / union) if union > 0 else 0.0

# =========================================================
# メイン処理
# =========================================================
if run_btn:
    with st.spinner("クラスタリング実行中..."):
        df_raw = load_data(region_option)
        df, y_true, y_pred, labels = run_clustering(df_raw, k, method)

    # --- 指標計算 ---
    acc  = accuracy_score(y_true, y_pred)
    iou  = calc_iou(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp_val, fn_val, tp_val = cm.ravel()

    # =========================================================
    # 指標表示
    # =========================================================
    st.subheader("📊 精度指標")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("IoU",      f"{iou:.4f}")
    c3.metric("Precision", f"{prec:.4f}")
    c4.metric("Recall",   f"{rec:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")

    # 混同行列
    st.subheader("🔢 混同行列")
    cm_cols = st.columns(4)
    cm_cols[0].metric("TP（正解：水田）",   f"{tp_val:,}")
    cm_cols[1].metric("FP（誤検知）",       f"{fp_val:,}", delta=f"-{fp_val:,}", delta_color="inverse")
    cm_cols[2].metric("FN（見逃し）",       f"{fn_val:,}", delta=f"-{fn_val:,}", delta_color="inverse")
    cm_cols[3].metric("TN（正解：非水田）", f"{tn:,}")

    st.divider()

    # =========================================================
    # 散布図 & 地図（横並び）
    # =========================================================
    col_scatter, col_map = st.columns(2)

    # --- 散布図 ---
    with col_scatter:
        st.subheader("🔵 散布図（VH_min × NDVI_grow）")
        fig, ax = plt.subplots(figsize=(6, 5))

        cmap = plt.get_cmap("tab10")
        for c in range(k):
            mask = df["cluster"] == c
            color = cmap(c / max(k - 1, 1))
            ax.scatter(
                df.loc[mask, "VH_min"],
                df.loc[mask, "NDVI_grow"],
                c=[color], s=3, alpha=0.4,
                label=f"クラスタ{c}"
            )

        ax.set_xlabel("VH_min（dB）", fontsize=11)
        ax.set_ylabel("NDVI_grow", fontsize=11)
        ax.set_title(f"{region_option}  {method}(k={k})", fontsize=12)
        ax.legend(markerscale=3, fontsize=8, loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- 地図 ---
    with col_map:
        st.subheader("🗺️ 地図表示（予測結果）")

        df_map = df.dropna(subset=["point_lat", "point_lng"]).copy()

        if df_map.empty:
            st.warning("緯度・経度データがありません")
        else:
            # サンプリング（地図が重くなりすぎないよう最大5000点）
            if len(df_map) > 5000:
                df_map = df_map.sample(5000, random_state=RANDOM_STATE)

            center_lat = df_map["point_lat"].mean()
            center_lng = df_map["point_lng"].mean()

            m = folium.Map(
                location=[center_lat, center_lng],
                zoom_start=12,
                tiles="CartoDB positron"
            )

            # 色設定
            color_map = {
                "TP": "blue",    # 正解水田
                "TN": "lightgray",  # 正解非水田
                "FP": "red",     # 誤検知（非水田を水田と判定）
                "FN": "orange",  # 見逃し（水田を非水田と判定）
            }
            label_map = {
                "TP": "正解（水田）",
                "TN": "正解（非水田）",
                "FP": "誤検知（FP）",
                "FN": "見逃し（FN）",
            }

            for _, row in df_map.iterrows():
                cls = row["分類"]
                folium.CircleMarker(
                    location=[row["point_lat"], row["point_lng"]],
                    radius=3,
                    color=color_map[cls],
                    fill=True,
                    fill_color=color_map[cls],
                    fill_opacity=0.7,
                    popup=f"{label_map[cls]}<br>VH_min: {row['VH_min']:.2f}<br>NDVI_grow: {row['NDVI_grow']:.3f}",
                ).add_to(m)

            # 凡例
            legend_html = """
            <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                        background: white; padding: 10px; border-radius: 8px;
                        border: 1px solid #ccc; font-size: 13px;">
                <b>凡例</b><br>
                🔵 正解（水田）<br>
                🔴 誤検知（FP）<br>
                🟠 見逃し（FN）<br>
                ⚪ 正解（非水田）
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

            st_folium(m, width=600, height=500)

    # サンプル数情報
    st.caption(
        f"サンプル数: {len(df):,}  ／  水田比率: {y_true.mean():.3f}"
        + (f"  ／  地図表示: 最大5,000点" if len(df_map) >= 5000 else "")
    )

else:
    st.info("👈 サイドバーで条件を設定して「▶ 実行」を押してください")
    st.markdown("""
    ### 使い方
    1. **地域**を選択（つくばみらい市・稲敷市・笠間市・全地域）
    2. **クラスタ数 k** をスライダーで調整
    3. **手法**を選択（K-Means または GMM）
    4. **▶ 実行** ボタンを押す

    ### 精度の目安（ベースライン）
    | 地域 | IoU |
    |------|-----|
    | つくばみらい市 | 80.1% |
    | 稲敷市 | 89.3% |
    | 笠間市 | 70.1% |
    """)
