"""
放棄地候補スコアリング → KML/CSV生成スクリプト（特徴量ベース）

特徴量CSVから「放棄地らしい」非水田農地を抽出する。
スコアリング基準（planドキュメントの実測値より）:
  放棄地: VH_min=-19.0, NDVI_flood=0.59, NDVI_grow=0.66, dNDVI=0.07, dVH=-0.0
  畑:     VH_min=-20.7, NDVI_flood=0.44, NDVI_grow=0.51, dNDVI=0.07, dVH=-0.3

放棄地の特徴:
  - VH_min が高め（-20 以上）← 草地化で水分少ない
  - NDVI_flood が高め（0.55 以上）← 湛水期も草が生えている
  - NDVI_grow が中〜高（0.55〜0.75）← 生育期も草で緑だが稲ほどではない
  - dVH が 0 に近い（季節変化が少ない）
  - K-Means Stage1 で非水田と判定されたもの

入力:
  - data/tsukubamirai_features.csv
  - labels/combined_labels.csv（既ラベル済み除外）

出力:
  - Downloads/つくばみらい市_abandoned_candidates_<N>.kml
  - Downloads/つくばみらい市_abandoned_candidates_<N>_labels.csv
"""

import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---- 設定 ----
FEATURES_CSV  = "data/tsukubamirai_features.csv"
LABELS_CSV    = "labels/combined_labels.csv"
REGION        = "つくばみらい市"
OUTPUT_DIR    = Path.home() / "Downloads"

STAGE1_K      = 4
BASE_FEATURES = ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"]
RANDOM_STATE  = 42
MATCH_THR     = 0.001  # 度（既ラベル済み除外用）

# 放棄地の典型値
ABD_VH_MIN      = -19.0
ABD_NDVI_FLOOD  = 0.59
ABD_NDVI_GROW   = 0.66
ABD_DVH_TARGET  = 0.0


def run_stage1_kmeans(df):
    X = df[BASE_FEATURES].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=STAGE1_K, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    centers_orig = scaler.inverse_transform(km.cluster_centers_)
    rice_cluster = int(np.argmin(centers_orig[:, BASE_FEATURES.index("VH_min")]))
    rice_mask = pd.Series(labels == rice_cluster, index=X.index)
    print(f"  [Stage1] 水田クラスタ={rice_cluster}, 水田={rice_mask.sum()}件 / {len(X)}件")
    return rice_mask


def load_labeled_coords(labels_csv, region):
    df = pd.read_csv(labels_csv, encoding="utf-8-sig")
    df_r = df[df["地域"] == region] if "地域" in df.columns else df
    coords = df_r[["lat", "lon"]].dropna().values
    print(f"  [既ラベル済み] {region}: {len(coords)}件 除外対象")
    return coords


def calc_abandoned_score(row):
    """放棄地らしさスコア（0〜100）"""
    score = 0.0

    # VH_min: -19.0 に近いほど高得点（水田=-22.8, 畑=-20.7, 放棄地=-19.0）
    vh_diff = abs(row["VH_min"] - ABD_VH_MIN)
    if vh_diff <= 1.0:
        score += 35
    elif vh_diff <= 2.0:
        score += 22
    elif vh_diff <= 3.0:
        score += 10
    elif row["VH_min"] < -22.0:  # 水田寄り → 減点
        score -= 15

    # NDVI_flood: 高め（0.55以上）が放棄地らしい（草地化）
    nf = row["NDVI_flood"]
    if nf >= 0.60:
        score += 30
    elif nf >= 0.50:
        score += 20
    elif nf >= 0.40:
        score += 8
    elif nf < 0.30:  # 低すぎ → 裸地（畑寄り）
        score -= 10

    # NDVI_grow: 0.55〜0.75 が放棄地らしい（稲=0.70, 畑=0.51）
    ng = row["NDVI_grow"]
    if 0.58 <= ng <= 0.75:
        score += 20
    elif 0.50 <= ng < 0.58:
        score += 10
    elif ng > 0.80:  # 高すぎ → 水田寄り
        score -= 5

    # dVH: 0 に近いほど放棄地らしい（水田=-3.9, 畑=-0.3, 放棄地=-0.0）
    dvh = row.get("dVH", row["VH_min"] - row["VH_winter"])
    dvh_abs = abs(dvh)
    if dvh_abs <= 0.3:
        score += 15
    elif dvh_abs <= 0.8:
        score += 8
    elif dvh_abs >= 2.5:  # 水田寄り → 減点
        score -= 10

    return max(0.0, score)


def add_kml_style(doc, style_id, href):
    style = ET.SubElement(doc, "Style", id=style_id)
    icon_style = ET.SubElement(style, "IconStyle")
    ET.SubElement(icon_style, "scale").text = "1.0"
    icon = ET.SubElement(icon_style, "Icon")
    ET.SubElement(icon, "href").text = href


def main(limit=150):
    print("=" * 60)
    print(f"放棄地候補スコアリング: {REGION}")
    print("=" * 60)

    df = pd.read_csv(FEATURES_CSV)
    print(f"\n  特徴量CSV: {len(df)}件")

    if "dNDVI" not in df.columns:
        df["dNDVI"] = df["NDVI_grow"] - df["NDVI_flood"]
    if "dVH" not in df.columns:
        df["dVH"] = df["VH_min"] - df["VH_winter"]

    # Stage1 で水田除外
    df_valid = df.dropna(subset=BASE_FEATURES).copy()
    rice_mask = run_stage1_kmeans(df_valid)
    df_nonrice = df_valid[~rice_mask].copy()
    print(f"  非水田: {len(df_nonrice)}件")

    # 既ラベル済み除外
    labeled_coords = load_labeled_coords(LABELS_CSV, REGION)
    feat_coords = df_nonrice[["point_lat", "point_lng"]].values
    if len(labeled_coords) > 0:
        tree = cKDTree(labeled_coords)
        dists, _ = tree.query(feat_coords, k=1)
        already = dists < MATCH_THR
    else:
        already = np.zeros(len(feat_coords), dtype=bool)

    before = len(df_nonrice)
    df_nonrice = df_nonrice[~already].copy()
    print(f"  既ラベル済み除外: {before - len(df_nonrice)}件 → 残り{len(df_nonrice)}件")

    # 放棄地らしさスコア
    df_nonrice["aban_score"] = df_nonrice.apply(calc_abandoned_score, axis=1)
    df_cands = df_nonrice.sort_values("aban_score", ascending=False).head(limit).reset_index(drop=True)

    print(f"\n  スコア上位{limit}件を候補として出力")
    print(f"  スコア: min={df_cands['aban_score'].min():.1f} "
          f"/ median={df_cands['aban_score'].median():.1f} "
          f"/ max={df_cands['aban_score'].max():.1f}")

    # KML / CSV 出力
    kml_path = OUTPUT_DIR / f"{REGION}_abandoned_candidates_{limit}.kml"
    csv_path = OUTPUT_DIR / f"{REGION}_abandoned_candidates_{limit}_labels.csv"

    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = f"{REGION} 放棄地候補 top{limit}"

    add_kml_style(doc, "high",   "http://maps.google.com/mapfiles/kml/paddle/red-circle.png")
    add_kml_style(doc, "medium", "http://maps.google.com/mapfiles/kml/paddle/orange-circle.png")
    add_kml_style(doc, "low",    "http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png")

    third = limit // 3
    folders = {}
    for name, label in [
        ("high",   f"高スコア（放棄地らしさ強） 1-{third}件"),
        ("medium", f"中スコア {third+1}-{third*2}件"),
        ("low",    f"低スコア {third*2+1}-{limit}件"),
    ]:
        folder = ET.SubElement(doc, "Folder")
        ET.SubElement(folder, "name").text = label
        folders[name] = folder

    csv_rows = []
    for rank, row in df_cands.iterrows():
        pin_id = f"A{rank+1:04d}"
        lat = row["point_lat"]
        lon = row["point_lng"]
        score = row["aban_score"]

        if rank < third:
            priority = "high"
        elif rank < third * 2:
            priority = "medium"
        else:
            priority = "low"

        pm = ET.SubElement(folders[priority], "Placemark")
        ET.SubElement(pm, "name").text = pin_id
        ET.SubElement(pm, "description").text = (
            f"放棄地スコア: {score:.1f}\n"
            f"VH_min: {row['VH_min']:.2f}\n"
            f"NDVI_flood: {row['NDVI_flood']:.3f}\n"
            f"NDVI_grow: {row['NDVI_grow']:.3f}\n"
            f"dVH: {row['dVH']:.3f}\n"
            f"lat: {lat:.6f}, lon: {lon:.6f}"
        )
        ET.SubElement(pm, "styleUrl").text = f"#{priority}"
        point = ET.SubElement(pm, "Point")
        ET.SubElement(point, "coordinates").text = f"{lon},{lat},0"

        csv_rows.append({
            "id":        pin_id,
            "lat":       f"{lat:.6f}",
            "lon":       f"{lon:.6f}",
            "遊休農地フラグ": "",
            "面積m2":    "",
            "住所":      "",
            "候補スコア": f"{score:.1f}",
            "ラベル":    "",
            "確信度":    "",
            "メモ":      "",
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
    print("  1. KMLをGoogle Earthで開く（赤=高スコア・橙=中・黄=低）")
    print("  2. ピンをクリックして放棄地かどうか目視確認")
    print("  3. CSVの「ラベル」列に 耕作中 / 放棄地 / 除外 を入力")
    print("  4. combined_labels.csv に追記")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    main(limit)
