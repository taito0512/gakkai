"""
畑候補スコアリング → KML/CSV生成スクリプト

「畑らしい」非水田農地を特徴量ベースで抽出する。
スコアリング基準:
  - VH_min が -22〜-17 の範囲（水田は低すぎ、放棄地は高め）
  - NDVI_grow が低め（畑は稲より低い: 0.35〜0.60）
  - dNDVI（= NDVI_grow - NDVI_flood）が小さい（畑は季節振幅が小さい）
  - dVH（= VH_min - VH_winter）の絶対値が小さい（畑はVH季節変化が少ない）
  - K-Means Stage1 で水田と判定されたものは除外

入力:
  - data/tsukubamirai_features.csv（特徴量・geometry付き）
  - labels/combined_labels.csv（既ラベル済み座標を除外するため）

出力:
  - Downloads/つくばみらい市_farm_candidates_<N>.kml
  - Downloads/つくばみらい市_farm_candidates_<N>_labels.csv
"""

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MATCH_THRESHOLD = 0.001  # 度（約100m）：この距離以内なら同一筆とみなす

# ---- 設定 ----
FEATURES_CSV    = "data/tsukubamirai_features.csv"
LABELS_CSV      = "labels/combined_labels.csv"
REGION          = "つくばみらい市"
OUTPUT_DIR      = Path.home() / "Downloads"

# K-Means Stage1 設定（accuracy_check.py と同一）
STAGE1_K        = 4
BASE_FEATURES   = ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"]
RANDOM_STATE    = 42
COORD_THR       = 0.001   # 度（既ラベル済み除外用）

# 畑らしさスコア用の目標値（planドキュメントの実測値より）
# 畑: VH_min=-20.7, NDVI_grow=0.51, dNDVI=0.07, dVH=-0.3
FARM_VH_MIN_CENTER    = -20.7
FARM_NDVI_GROW_CENTER = 0.51
FARM_DNDVI_TARGET     = 0.07   # 小さいほど畑らしい
FARM_DVH_TARGET       = -0.3   # 0に近いほど畑らしい


def run_stage1_kmeans(df: pd.DataFrame) -> pd.Series:
    """
    K-Means (k=4) で水田クラスタを同定し、水田と判定された行のインデックスを返す。
    水田クラスタ = VH_min が最小のクラスタ（第1段階と同じ判定ロジック）。
    """
    X = df[BASE_FEATURES].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=STAGE1_K, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)

    # VH_min が最小のクラスタ → 水田
    centers_orig = scaler.inverse_transform(km.cluster_centers_)
    rice_cluster = int(np.argmin(centers_orig[:, BASE_FEATURES.index("VH_min")]))

    rice_mask = pd.Series(labels == rice_cluster, index=X.index)
    n_rice = rice_mask.sum()
    print(f"  [Stage1 K-Means] 水田クラスタ={rice_cluster}, 水田と判定={n_rice}件 / {len(X)}件")
    return rice_mask


def load_labeled_coords(labels_csv: str, region: str):
    """combined_labels.csv から指定地域の既ラベル済み座標配列を返す（KDTree用）。"""
    df = pd.read_csv(labels_csv, encoding="utf-8-sig")
    df_region = df[df["地域"] == region] if "地域" in df.columns else df
    coords = df_region[["lat", "lon"]].dropna().values
    print(f"  [既ラベル済み] {region}: {len(coords)}件 除外対象")
    return coords


def calc_farm_score(row: pd.Series) -> float:
    """
    畑らしさスコア（0〜100）を計算する。
    各特徴量が「畑の典型値」に近いほど高スコア。
    """
    score = 0.0

    # VH_min: 畑の中心 -20.7 に近いほど高得点（±3dB 以内で加点）
    vh_diff = abs(row["VH_min"] - FARM_VH_MIN_CENTER)
    if vh_diff <= 1.0:
        score += 35
    elif vh_diff <= 2.0:
        score += 25
    elif vh_diff <= 3.0:
        score += 15
    elif vh_diff <= 5.0:
        score += 5

    # NDVI_grow: 畑の中心 0.51 に近いほど高得点（稲は0.70、放棄地は0.66）
    ndvi_diff = abs(row["NDVI_grow"] - FARM_NDVI_GROW_CENTER)
    if ndvi_diff <= 0.08:
        score += 30
    elif ndvi_diff <= 0.15:
        score += 20
    elif ndvi_diff <= 0.25:
        score += 10

    # dNDVI: 小さいほど畑らしい（水田は0.29と大きい）
    dndvi = row.get("dNDVI", row["NDVI_grow"] - row["NDVI_flood"])
    if dndvi <= 0.08:
        score += 20
    elif dndvi <= 0.12:
        score += 12
    elif dndvi <= 0.18:
        score += 5
    elif dndvi >= 0.22:  # 水田寄り → 減点
        score -= 10

    # dVH: 0 に近いほど畑らしい（水田は -3.9 と大きい）
    dvh = row.get("dVH", row["VH_min"] - row["VH_winter"])
    dvh_abs = abs(dvh)
    if dvh_abs <= 0.5:
        score += 15
    elif dvh_abs <= 1.0:
        score += 10
    elif dvh_abs <= 2.0:
        score += 3
    elif dvh_abs >= 3.0:  # 水田寄り → 減点
        score -= 10

    return max(0.0, score)


def add_kml_style(doc, style_id, href):
    style = ET.SubElement(doc, "Style", id=style_id)
    icon_style = ET.SubElement(style, "IconStyle")
    ET.SubElement(icon_style, "scale").text = "1.0"
    icon = ET.SubElement(icon_style, "Icon")
    ET.SubElement(icon, "href").text = href


def main(limit: int = 300):
    print("=" * 60)
    print(f"畑候補スコアリング: {REGION}")
    print("=" * 60)

    # ---- データ読み込み ----
    df = pd.read_csv(FEATURES_CSV)
    print(f"\n  特徴量CSV: {len(df)}件")

    # dNDVI / dVH を計算（なければ追加）
    if "dNDVI" not in df.columns:
        df["dNDVI"] = df["NDVI_grow"] - df["NDVI_flood"]
    if "dVH" not in df.columns:
        df["dVH"] = df["VH_min"] - df["VH_winter"]

    # ---- Stage1 で水田除外 ----
    df_valid = df.dropna(subset=BASE_FEATURES).copy()
    rice_mask = run_stage1_kmeans(df_valid)
    df_nonrice = df_valid[~rice_mask].copy()
    print(f"  非水田: {len(df_nonrice)}件")

    # ---- 既ラベル済み座標を除外（KDTree 近傍マッチング）----
    labeled_coords = load_labeled_coords(LABELS_CSV, REGION)
    feat_coords = df_nonrice[["point_lat", "point_lng"]].values
    if len(labeled_coords) > 0:
        tree = cKDTree(labeled_coords)
        dists, _ = tree.query(feat_coords, k=1)
        already_labeled = dists < MATCH_THRESHOLD
    else:
        already_labeled = np.zeros(len(feat_coords), dtype=bool)

    before = len(df_nonrice)
    df_nonrice = df_nonrice[~already_labeled].copy()
    print(f"  既ラベル済み除外: {before - len(df_nonrice)}件 → 残り{len(df_nonrice)}件")

    # ---- 畑らしさスコア計算 ----
    df_nonrice["farm_score"] = df_nonrice.apply(calc_farm_score, axis=1)
    df_cands = df_nonrice.sort_values("farm_score", ascending=False).head(limit).copy()
    df_cands = df_cands.reset_index(drop=True)

    print(f"\n  スコア上位{limit}件を候補として出力")
    print(f"  スコア分布: min={df_cands['farm_score'].min():.1f} "
          f"/ median={df_cands['farm_score'].median():.1f} "
          f"/ max={df_cands['farm_score'].max():.1f}")

    # ---- KML / CSV 出力 ----
    kml_path = OUTPUT_DIR / f"{REGION}_farm_candidates_{limit}.kml"
    csv_path = OUTPUT_DIR / f"{REGION}_farm_candidates_{limit}_labels.csv"

    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = f"{REGION} 畑候補 top{limit}"

    add_kml_style(doc, "high",   "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png")
    add_kml_style(doc, "medium", "http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png")
    add_kml_style(doc, "low",    "http://maps.google.com/mapfiles/kml/paddle/wht-circle.png")

    third = limit // 3
    folders = {}
    for name, label in [
        ("high",   f"高スコア（畑らしさ強） 1-{third}件"),
        ("medium", f"中スコア {third+1}-{third*2}件"),
        ("low",    f"低スコア {third*2+1}-{limit}件"),
    ]:
        folder = ET.SubElement(doc, "Folder")
        ET.SubElement(folder, "name").text = label
        folders[name] = folder

    csv_rows = []
    for rank, row in df_cands.iterrows():
        pin_id = f"F{rank+1:04d}"
        lat = row["point_lat"]
        lon = row["point_lng"]
        score = row["farm_score"]

        # スコアに応じてスタイル分け
        if rank < third:
            priority = "high"
        elif rank < third * 2:
            priority = "medium"
        else:
            priority = "low"

        # KML ピン
        pm = ET.SubElement(folders[priority], "Placemark")
        ET.SubElement(pm, "name").text = pin_id
        ET.SubElement(pm, "description").text = (
            f"畑スコア: {score:.1f}\n"
            f"VH_min: {row['VH_min']:.2f}\n"
            f"NDVI_grow: {row['NDVI_grow']:.3f}\n"
            f"dNDVI: {row['dNDVI']:.3f}\n"
            f"dVH: {row['dVH']:.3f}\n"
            f"lat: {lat:.6f}, lon: {lon:.6f}"
        )
        ET.SubElement(pm, "styleUrl").text = f"#{priority}"
        point = ET.SubElement(pm, "Point")
        ET.SubElement(point, "coordinates").text = f"{lon},{lat},0"

        # CSV 行（combined_labels.csv に追加できる形式）
        csv_rows.append({
            "id":           pin_id,
            "lat":          f"{lat:.6f}",
            "lon":          f"{lon:.6f}",
            "遊休農地フラグ": "",
            "面積m2":       "",
            "住所":         "",
            "候補スコア":   f"{score:.1f}",
            "ラベル":       "",   # ← 目視ラベリング入力欄
            "確信度":       "",   # ← 高/中
            "メモ":         "",
        })

    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    tree.write(kml_path, encoding="utf-8", xml_declaration=True)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["id", "lat", "lon", "遊休農地フラグ", "面積m2", "住所",
                      "候補スコア", "ラベル", "確信度", "メモ"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n  KML → {kml_path}")
    print(f"  CSV → {csv_path}")
    print("\n手順:")
    print("  1. KMLをGoogle Earthで開く（緑=高スコア・黄=中・白=低）")
    print("  2. ピンをクリックして畑かどうか目視確認")
    print("  3. CSVの「ラベル」列に 耕作中 / 放棄地 / 除外 を入力")
    print("  4. combined_labels.csv に追記")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    main(limit)
