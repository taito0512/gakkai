"""
十日町市 放棄地候補 空間クラスタリングKML生成スクリプト

目的:
  放棄地らしい筆を空間的にまとめてKMLに出力し、
  1枚のスクリーンショットで複数ピンを目視確認できるようにする。

処理:
  1. 筆ポリゴンshp + v2特徴量CSV を結合（polygon_uu）
  2. 既にラベル済みの150件を除外
  3. 放棄地らしさスコアを計算
       - NDVI_apr 高い（放棄後は雑草が増え春NDVIが高い）
       - NDWI_mean 低い（水分少ない）
       - VH_mean 中程度（水田より高い）
  4. ~0.5kmグリッド（0.005°セル）で空間クラスタリング
  5. スコア上位グリッドセルを優先し、各セル最大N件選択
  6. KML + CSV出力

入力:
  ~/Downloads/2023_tokamachi/2023_tokamachi.shp
  {GDRIVE}/tokamachi_features_v2_2023.csv
  ~/Downloads/十日町市_labeling_150_labels.csv

出力:
  ~/Downloads/十日町市_abandoned_candidates_clustered_<N>.kml
  ~/Downloads/十日町市_abandoned_candidates_clustered_<N>_labels.csv

使い方:
  python scripts/create_abandoned_candidate_kml_tokamachi.py        # 200件
  python scripts/create_abandoned_candidate_kml_tokamachi.py 100    # 100件
"""

import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ---- パス設定 ----
OUTPUT_DIR   = Path.home() / "Downloads"
SHP_PATH     = Path.home() / "Downloads/2023_tokamachi/2023_tokamachi.shp"
LABELED_CSV  = Path.home() / "Downloads/十日町市_labeling_150_labels.csv"
RANDOM_STATE = 42

_GDRIVE  = (
    "/Users/nagatomotaito/Library/CloudStorage"
    "/GoogleDrive-ntaito.50@gmail.com/マイドライブ"
)
V2_CSV       = f"{_GDRIVE}/tokamachi_features_v2_2023.csv"
# 農振農用地ポリゴン（国土数値情報 A12 新潟県）
NOUSON_SHP   = Path.home() / "Downloads/A12-15_15_GML"

# グリッドセルサイズ（度）約0.5km
GRID_SIZE = 0.005
# 1セルから最大何件選ぶか
MAX_PER_CELL = 5
# スコア計算に使う特徴量の重み（正=高いほど放棄地らしい）
SCORE_WEIGHTS = {
    "NDVI_apr":  +1.5,   # 春の雑草・放棄後植生
    "NDWI_mean": -1.0,   # 湛水なし
    "VH_mean":   +0.5,   # 水田より後方散乱やや高い
}


def add_kml_style(doc, style_id, color_href):
    style = ET.SubElement(doc, "Style", id=style_id)
    icon_style = ET.SubElement(style, "IconStyle")
    ET.SubElement(icon_style, "scale").text = "1.0"
    icon = ET.SubElement(icon_style, "Icon")
    ET.SubElement(icon, "href").text = color_href


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


def main(n_total: int = 200):
    print("=" * 60)
    print("  十日町市 放棄地候補 空間クラスタリングKML生成")
    print("=" * 60)

    # ---- shp読み込み ----
    if not SHP_PATH.exists():
        print(f"  [ERROR] shpが見つかりません: {SHP_PATH}")
        return
    gdf = gpd.read_file(SHP_PATH)
    print(f"  全筆数: {len(gdf):,}件")

    # 畑（land_type=200）のみ対象
    gdf = gdf[gdf["land_type"] == 200].copy()
    print(f"  畑（land_type=200）: {len(gdf):,}件")

    # ---- v2特徴量CSV読み込み・結合 ----
    v2_path = Path(V2_CSV)
    if not v2_path.exists():
        print(f"  [ERROR] v2 CSVが見つかりません: {V2_CSV}")
        return
    v2 = pd.read_csv(v2_path)
    print(f"  v2特徴量CSV: {len(v2):,}件")

    df = gdf[["polygon_uu", "point_lat", "point_lng"]].copy()
    df = df.merge(v2, on="polygon_uu", how="inner")
    print(f"  JOIN後: {len(df):,}件")

    # ---- 農振農用地フィルタ（国土数値情報 A12） ----
    # 農用地区域内の筆のみ対象とすることで森林・集落内を事前除去
    shp1 = NOUSON_SHP / "a001150020160205.shp"
    shp2 = NOUSON_SHP / "a001150020160206.shp"
    if shp1.exists() and shp2.exists():
        nouson = pd.concat([
            gpd.read_file(shp1),
            gpd.read_file(shp2),
        ], ignore_index=True).to_crs("EPSG:4326")
        nouson_union = nouson.geometry.union_all()

        # 筆の代表点が農振ポリゴン内に含まれるものだけ残す
        from shapely.geometry import Point
        before = len(df)
        mask = df.apply(
            lambda r: nouson_union.contains(Point(r["point_lng"], r["point_lat"])),
            axis=1
        )
        df = df[mask].copy()
        print(f"  農振農用地フィルタ後: {before - len(df)}件除去 → 残り{len(df):,}件")
    else:
        print("  [WARNING] 農振shpが見つかりません。フィルタをスキップします。")

    # ---- 既ラベル済みを除外（ラベル入力済みのもののみ） ----
    if LABELED_CSV.exists():
        labeled = pd.read_csv(LABELED_CSV, encoding="utf-8-sig")
        labeled_uu = set(
            labeled[labeled["ラベル"].notna()]["polygon_uu"].dropna().astype(str)
        )
        if labeled_uu:
            before = len(df)
            df = df[~df["polygon_uu"].astype(str).isin(labeled_uu)].copy()
            print(f"  既ラベル除外: {before - len(df)}件除外 → 残り{len(df):,}件")
        else:
            print("  既ラベルなし（全件対象）")
    else:
        print(f"  [WARNING] ラベル済みCSVが見つかりません: {LABELED_CSV}")

    # ---- 放棄地らしさスコア計算 ----
    df["_score"] = compute_score(df)

    # ---- 空間グリッドクラスタリング ----
    df["_grid_lat"] = (df["point_lat"] / GRID_SIZE).astype(int)
    df["_grid_lng"] = (df["point_lng"] / GRID_SIZE).astype(int)
    df["_grid_id"]  = df["_grid_lat"].astype(str) + "_" + df["_grid_lng"].astype(str)

    # グリッドごとのスコア集計
    grid_stats = (
        df.groupby("_grid_id")["_score"]
        .agg(count="count", mean_score="mean")
        .reset_index()
    )
    # スコア平均が高く件数も多いセルを優先
    grid_stats["_grid_priority"] = grid_stats["mean_score"] * np.log1p(grid_stats["count"])
    grid_stats = grid_stats.sort_values("_grid_priority", ascending=False)

    # 上位グリッドセルから順にMAX_PER_CELL件ずつ収集
    selected_rows = []
    for gid in grid_stats["_grid_id"]:
        if len(selected_rows) >= n_total:
            break
        cell = df[df["_grid_id"] == gid].nlargest(MAX_PER_CELL, "_score")
        selected_rows.append(cell)

    selected = pd.concat(selected_rows, ignore_index=True).head(n_total)
    print(f"  選択件数: {len(selected)}件  使用グリッドセル: {selected['_grid_id'].nunique()}セル")

    # ---- KML生成 ----
    n = len(selected)
    kml_path = OUTPUT_DIR / f"十日町市_abandoned_candidates_clustered_{n}.kml"
    csv_path = OUTPUT_DIR / f"十日町市_abandoned_candidates_clustered_{n}_labels.csv"

    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = "十日町市 放棄地候補（空間クラスタリング）"

    add_kml_style(doc, "pin_yellow",
                  "http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png")
    add_kml_style(doc, "pin_green",
                  "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png")
    add_kml_style(doc, "pin_red",
                  "http://maps.google.com/mapfiles/kml/paddle/red-circle.png")

    folder = ET.SubElement(doc, "Folder")
    ET.SubElement(folder, "name").text = f"放棄地候補（クラスタ優先{n}件）"

    csv_rows = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        pin_id  = f"TKA{rank:04d}"
        lat     = float(row["point_lat"])
        lon     = float(row["point_lng"])
        poly_uu = str(row["polygon_uu"])
        score   = float(row["_score"])
        grid_id = str(row["_grid_id"])

        pm = ET.SubElement(folder, "Placemark")
        ET.SubElement(pm, "name").text = pin_id
        ET.SubElement(pm, "description").text = (
            f"polygon_uu: {poly_uu}\n"
            f"score: {score:.3f}\n"
            f"grid: {grid_id}\n"
            f"NDVI_apr: {row.get('NDVI_apr', 'N/A')}\n"
            f"NDWI_mean: {row.get('NDWI_mean', 'N/A')}\n"
            f"VH_mean: {row.get('VH_mean', 'N/A')}\n"
            f"lat: {lat:.6f}, lon: {lon:.6f}"
        )
        ET.SubElement(pm, "styleUrl").text = "#pin_yellow"
        point = ET.SubElement(pm, "Point")
        ET.SubElement(point, "coordinates").text = f"{lon},{lat},0"

        csv_rows.append({
            "地域":          "十日町市",
            "id":            pin_id,
            "lat":           f"{lat:.6f}",
            "lon":           f"{lon:.6f}",
            "polygon_uu":    poly_uu,
            "score":         f"{score:.3f}",
            "grid_id":       grid_id,
            "NDVI_apr":      row.get("NDVI_apr", ""),
            "NDWI_mean":     row.get("NDWI_mean", ""),
            "VH_mean":       row.get("VH_mean", ""),
            "遊休農地フラグ": "",
            "ラベル":        "",   # 耕作中 / 放棄地 / 除外
            "確信度":        "",   # 高 / 中
            "メモ":          "",
        })

    tree_obj = ET.ElementTree(kml)
    ET.indent(tree_obj, space="  ")
    tree_obj.write(kml_path, encoding="utf-8", xml_declaration=True)

    fieldnames = ["地域", "id", "lat", "lon", "polygon_uu", "score", "grid_id",
                  "NDVI_apr", "NDWI_mean", "VH_mean",
                  "遊休農地フラグ", "ラベル", "確信度", "メモ"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n  KML → {kml_path}")
    print(f"  CSV → {csv_path}")
    print("\n  手順:")
    print("    1. KMLをGoogle Earthで開く")
    print("    2. 同じグリッドエリアのピンが近くに固まっているので1枚でスクショ可能")
    print("    3. CSVの「ラベル」列に 耕作中 / 放棄地 / 除外 を入力")
    print("    4. combined_labels.csv に追記")
    print("\n完了")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(n)
