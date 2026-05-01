"""
香取市 追加ラベリング用KML生成スクリプト（v2）

目的:
  放棄地候補150件＋耕作中候補80件を1つのKMLに出力して
  効率よくラベリングできるようにする。

変更点（v2）:
  - シェープファイル不要 → base CSV（座標・基本特徴量）を使用
  - 既ラベル済み除外を combined_labels.csv から行う
  - 放棄地候補（赤）+ 耕作中候補（青）を1KMLに統合

入力:
  {GEE_EXPORTS}/katori_features_base_2023.csv   （座標 + 基本特徴量）
  {GEE_EXPORTS}/katori_features_v2_2023.csv     （NDVI_apr・NDWI_mean等）
  labels/combined_labels.csv                    （既ラベル済み除外用）

出力:
  ~/Downloads/香取市_labeling_v2_<ABD>abd_<ACT>act.kml
  ~/Downloads/香取市_labeling_v2_<ABD>abd_<ACT>act_labels.csv

使い方:
  python scripts/create_abandoned_candidate_kml_katori.py           # 150+80
  python scripts/create_abandoned_candidate_kml_katori.py 100 50    # 100+50
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

# ---- パス設定 ----
OUTPUT_DIR   = Path.home() / "Downloads"
LABELS_CSV   = Path("labels/combined_labels.csv")
RANDOM_STATE = 42

_GDRIVE  = (
    "/Users/nagatomotaito/Library/CloudStorage"
    "/GoogleDrive-ntaito.50@gmail.com/マイドライブ"
)
_GEE     = f"{_GDRIVE}/GEE_exports"
BASE_CSV = f"{_GEE}/katori_features_base_2023.csv"
V2_CSV   = f"{_GEE}/katori_features_v2_2023.csv"

# グリッドセルサイズ（度）約0.5km
GRID_SIZE    = 0.005
MAX_PER_CELL = 5

# 放棄地らしさスコアの重み（z標準化後）
SCORE_WEIGHTS = {
    "NDVI_apr":  +1.5,   # 春の雑草・放棄後植生
    "NDWI_mean": -1.0,   # 湛水なし
    "VH_mean":   +0.5,   # 水田より後方散乱やや高い
}

# 第1段階K-Meansで使う特徴量（水田を除外するため）
STAGE1_FEATURES = ["VH_min", "NDVI_grow", "VH_winter", "NDVI_flood"]


def add_kml_style(doc, style_id, href, scale="1.0"):
    style = ET.SubElement(doc, "Style", id=style_id)
    icon_style = ET.SubElement(style, "IconStyle")
    ET.SubElement(icon_style, "scale").text = scale
    icon = ET.SubElement(icon_style, "Icon")
    ET.SubElement(icon, "href").text = href


def compute_score(df: pd.DataFrame) -> pd.Series:
    """各特徴量をz標準化してウェイト付き和でスコア計算"""
    score = pd.Series(0.0, index=df.index)
    for col, w in SCORE_WEIGHTS.items():
        if col not in df.columns:
            continue
        vals = df[col].astype(float)
        std  = vals.std()
        if std > 0:
            score += w * (vals - vals.mean()) / std
    return score


def exclude_water_fields(df: pd.DataFrame) -> pd.DataFrame:
    """K-Means Stage1で水田を除外（VH_minが最小のクラスタを水田と判定）"""
    valid = df.dropna(subset=STAGE1_FEATURES)
    scaler = StandardScaler()
    X = scaler.fit_transform(valid[STAGE1_FEATURES])
    km = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    centers = scaler.inverse_transform(km.cluster_centers_)
    rice_cluster = int(np.argmin(centers[:, STAGE1_FEATURES.index("VH_min")]))
    rice_mask = labels == rice_cluster
    nonrice_idx = valid.index[~rice_mask]
    print(f"  [Stage1] 水田クラスタ={rice_cluster}: "
          f"水田{rice_mask.sum():,}件除去 → 非水田{len(nonrice_idx):,}件")
    return df.loc[nonrice_idx].copy()


def spatial_cluster_select(df: pd.DataFrame, n: int, score_col: str,
                            ascending: bool = False) -> pd.DataFrame:
    """空間グリッドクラスタリングでn件選択（ascending=Falseで高スコア優先）"""
    df = df.copy()
    df["_grid_id"] = (
        (df["point_lat"] / GRID_SIZE).astype(int).astype(str) + "_" +
        (df["point_lng"] / GRID_SIZE).astype(int).astype(str)
    )
    agg_fn = "mean" if not ascending else "mean"
    grid_stats = (
        df.groupby("_grid_id")[score_col]
        .agg(count="count", mean_score="mean")
        .reset_index()
    )
    if ascending:
        grid_stats["_priority"] = -grid_stats["mean_score"] * np.log1p(grid_stats["count"])
    else:
        grid_stats["_priority"] = grid_stats["mean_score"] * np.log1p(grid_stats["count"])
    grid_stats = grid_stats.sort_values("_priority", ascending=False)

    rows = []
    for gid in grid_stats["_grid_id"]:
        if len(rows) >= n:
            break
        cell = df[df["_grid_id"] == gid]
        if ascending:
            cell = cell.nsmallest(MAX_PER_CELL, score_col)
        else:
            cell = cell.nlargest(MAX_PER_CELL, score_col)
        rows.append(cell)
    return pd.concat(rows, ignore_index=True).head(n)


def write_kml_placemark(folder, pin_id, lat, lon, score, row, style):
    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = pin_id
    ET.SubElement(pm, "description").text = (
        f"polygon_uu: {row['polygon_uu']}\n"
        f"score: {score:.3f}\n"
        f"NDVI_apr: {row.get('NDVI_apr', 'N/A'):.3f}\n"
        f"NDWI_mean: {row.get('NDWI_mean', 'N/A'):.3f}\n"
        f"VH_mean: {row.get('VH_mean', 'N/A'):.2f}\n"
        f"lat: {lat:.6f}, lon: {lon:.6f}"
    )
    ET.SubElement(pm, "styleUrl").text = f"#{style}"
    point = ET.SubElement(pm, "Point")
    ET.SubElement(point, "coordinates").text = f"{lon},{lat},0"


def main(n_abd: int = 150, n_act: int = 80):
    print("=" * 60)
    print("  香取市 追加ラベリング用KML生成（v2）")
    print(f"  目標: 放棄地候補{n_abd}件 + 耕作中候補{n_act}件")
    print("=" * 60)

    # ---- 特徴量読み込み ----
    base = pd.read_csv(BASE_CSV)
    v2   = pd.read_csv(V2_CSV)
    df   = base.merge(v2, on="polygon_uu", how="inner")
    print(f"  base({len(base):,}) × v2({len(v2):,}) → JOIN後: {len(df):,}件")

    # ---- 水田除外（Stage1 K-Means） ----
    df = exclude_water_fields(df)

    # ---- 既ラベル済み除外（combined_labels.csv の香取市・lat/lon近傍） ----
    if LABELS_CSV.exists():
        ldf = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
        katori_labels = ldf[ldf["地域"] == "香取市"][["lat", "lon"]].dropna()
        if len(katori_labels) > 0:
            tree = cKDTree(katori_labels.values)
            dists, _ = tree.query(df[["point_lat", "point_lng"]].values, k=1)
            already = dists < 0.001  # 約100m以内は既ラベル済みとみなす
            before = len(df)
            df = df[~already].copy()
            print(f"  既ラベル除外: {before - len(df)}件 → 残り{len(df):,}件")

    # ---- スコア計算 ----
    df["_score"] = compute_score(df)
    print(f"  スコア範囲: {df['_score'].min():.2f} ～ {df['_score'].max():.2f}")

    # ---- 放棄地候補（高スコア）・耕作中候補（低スコア）選択 ----
    abd_cands = spatial_cluster_select(df, n_abd, "_score", ascending=False)
    # 耕作中候補は放棄地候補と重複しないように除外
    df_rest   = df[~df["polygon_uu"].isin(abd_cands["polygon_uu"])].copy()
    act_cands = spatial_cluster_select(df_rest, n_act, "_score", ascending=True)

    print(f"  放棄地候補: {len(abd_cands)}件  "
          f"スコア中央値={abd_cands['_score'].median():.2f}")
    print(f"  耕作中候補: {len(act_cands)}件  "
          f"スコア中央値={act_cands['_score'].median():.2f}")

    # ---- KML生成 ----
    tag = f"{len(abd_cands)}abd_{len(act_cands)}act"
    kml_path = OUTPUT_DIR / f"香取市_labeling_v2_{tag}.kml"
    csv_path = OUTPUT_DIR / f"香取市_labeling_v2_{tag}_labels.csv"

    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = "香取市 追加ラベリング（v2）"

    add_kml_style(doc, "pin_red",
                  "http://maps.google.com/mapfiles/kml/paddle/red-circle.png")
    add_kml_style(doc, "pin_blue",
                  "http://maps.google.com/mapfiles/kml/paddle/blu-circle.png")

    folder_abd = ET.SubElement(doc, "Folder")
    ET.SubElement(folder_abd, "name").text = f"放棄地候補（赤・{len(abd_cands)}件）"
    folder_act = ET.SubElement(doc, "Folder")
    ET.SubElement(folder_act, "name").text = f"耕作中候補（青・{len(act_cands)}件）"

    csv_rows = []
    for rank, (_, row) in enumerate(abd_cands.iterrows(), start=1):
        pin_id = f"KTA_A{rank:04d}"
        lat, lon = float(row["point_lat"]), float(row["point_lng"])
        write_kml_placemark(folder_abd, pin_id, lat, lon, row["_score"], row, "pin_red")
        csv_rows.append({
            "地域": "香取市", "id": pin_id,
            "lat": f"{lat:.6f}", "lon": f"{lon:.6f}",
            "polygon_uu": str(row["polygon_uu"]),
            "候補種別": "放棄地候補",
            "score": f"{row['_score']:.3f}",
            "遊休農地フラグ": "", "ラベル": "", "確信度": "", "メモ": "",
        })

    for rank, (_, row) in enumerate(act_cands.iterrows(), start=1):
        pin_id = f"KTA_C{rank:04d}"
        lat, lon = float(row["point_lat"]), float(row["point_lng"])
        write_kml_placemark(folder_act, pin_id, lat, lon, row["_score"], row, "pin_blue")
        csv_rows.append({
            "地域": "香取市", "id": pin_id,
            "lat": f"{lat:.6f}", "lon": f"{lon:.6f}",
            "polygon_uu": str(row["polygon_uu"]),
            "候補種別": "耕作中候補",
            "score": f"{row['_score']:.3f}",
            "遊休農地フラグ": "", "ラベル": "", "確信度": "", "メモ": "",
        })

    tree_obj = ET.ElementTree(kml)
    ET.indent(tree_obj, space="  ")
    tree_obj.write(kml_path, encoding="utf-8", xml_declaration=True)

    fieldnames = ["地域", "id", "lat", "lon", "polygon_uu", "候補種別",
                  "score", "遊休農地フラグ", "ラベル", "確信度", "メモ"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n  KML → {kml_path}")
    print(f"  CSV → {csv_path}")
    print("\n  手順:")
    print("    赤ピン = 放棄地候補（高スコア）")
    print("    青ピン = 耕作中候補（低スコア・畑らしい）")
    print("    CSVの「ラベル」列に 耕作中 / 放棄地 / 除外 を入力")
    print("    完了後 → python scripts/add_labels_to_combined.py で追記")
    print("\n完了")


if __name__ == "__main__":
    args = sys.argv[1:]
    n_abd = int(args[0]) if len(args) > 0 else 150
    n_act = int(args[1]) if len(args) > 1 else 80
    main(n_abd, n_act)
