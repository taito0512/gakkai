"""
香取市・十日町市 ラベリング用KML/CSV生成スクリプト

筆ポリゴンのshpファイルから非水田農地（land_type=200）をランダムサンプリングし、
Google Earthで目視確認できるKMLとラベル記録用CSVを生成する。

入力:
  ~/Downloads/2023_katori/2023_katori.shp
  ~/Downloads/2023_tokamachi/2023_tokamachi.shp

出力:
  ~/Downloads/香取市_labeling_<N>.kml
  ~/Downloads/香取市_labeling_<N>_labels.csv
  ~/Downloads/十日町市_labeling_<N>.kml
  ~/Downloads/十日町市_labeling_<N>_labels.csv

使い方:
  python scripts/create_labeling_kml_new_regions.py          # 両地域・各150件
  python scripts/create_labeling_kml_new_regions.py 80       # 両地域・各80件
"""

import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np

OUTPUT_DIR   = Path.home() / "Downloads"
RANDOM_STATE = 42

# 対象地域の設定
REGIONS = [
    {
        "name":    "香取市",
        "shp":     Path.home() / "Downloads/2023_katori/2023_katori.shp",
        "prefix":  "KT",   # KML ピンIDのプレフィックス
    },
    {
        "name":    "十日町市",
        "shp":     Path.home() / "Downloads/2023_tokamachi/2023_tokamachi.shp",
        "prefix":  "TK",
    },
]

# land_type の意味（農水省筆ポリゴン仕様）
# 100 = 田（水田・水稲作付が多い）
# 200 = 畑（非水田農地・ラベリング対象）
HATAKE_TYPE = 200


def add_kml_style(doc, style_id, color_href):
    style = ET.SubElement(doc, "Style", id=style_id)
    icon_style = ET.SubElement(style, "IconStyle")
    ET.SubElement(icon_style, "scale").text = "1.0"
    icon = ET.SubElement(icon_style, "Icon")
    ET.SubElement(icon, "href").text = color_href


def generate_kml_csv(region_cfg: dict, n_samples: int):
    name   = region_cfg["name"]
    shp    = region_cfg["shp"]
    prefix = region_cfg["prefix"]

    print(f"\n{'='*55}")
    print(f"  {name} ラベリング候補生成（{n_samples}件）")
    print(f"{'='*55}")

    if not shp.exists():
        print(f"  ⚠ shpが見つかりません: {shp}")
        return

    gdf = gpd.read_file(shp)
    print(f"  全筆数: {len(gdf):,}件")

    # 畑（land_type=200）のみ抽出
    hatake = gdf[gdf["land_type"] == HATAKE_TYPE].copy()
    print(f"  畑（land_type=200）: {len(hatake):,}件")

    # ランダムサンプリング
    n = min(n_samples, len(hatake))
    sampled = hatake.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  サンプリング: {n}件")

    # 出力パス
    kml_path = OUTPUT_DIR / f"{name}_labeling_{n}.kml"
    csv_path = OUTPUT_DIR / f"{name}_labeling_{n}_labels.csv"

    # ---- KML生成 ----
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = f"教師データ作成 - {name}"

    add_kml_style(doc, "pin_yellow",
                  "http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png")
    add_kml_style(doc, "pin_green",
                  "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png")
    add_kml_style(doc, "pin_red",
                  "http://maps.google.com/mapfiles/kml/paddle/red-circle.png")

    folder = ET.SubElement(doc, "Folder")
    ET.SubElement(folder, "name").text = f"畑候補（ランダム{n}件）"

    csv_rows = []
    for rank, row in sampled.iterrows():
        pin_id = f"{prefix}{rank+1:04d}"
        lat = float(row["point_lat"])
        lon = float(row["point_lng"])
        poly_uu = str(row["polygon_uu"])

        pm = ET.SubElement(folder, "Placemark")
        ET.SubElement(pm, "name").text = pin_id
        ET.SubElement(pm, "description").text = (
            f"polygon_uu: {poly_uu}\n"
            f"land_type: {row['land_type']}\n"
            f"lat: {lat:.6f}, lon: {lon:.6f}"
        )
        ET.SubElement(pm, "styleUrl").text = "#pin_yellow"
        point = ET.SubElement(pm, "Point")
        ET.SubElement(point, "coordinates").text = f"{lon},{lat},0"

        csv_rows.append({
            "地域":          name,
            "id":            pin_id,
            "lat":           f"{lat:.6f}",
            "lon":           f"{lon:.6f}",
            "polygon_uu":    poly_uu,
            "遊休農地フラグ": "",
            "area_m2":       "",
            "住所":          "",
            "ラベル":        "",   # 耕作中 / 放棄地 / 除外
            "確信度":        "",   # 高 / 中
            "メモ":          "",
        })

    tree_obj = ET.ElementTree(kml)
    ET.indent(tree_obj, space="  ")
    tree_obj.write(kml_path, encoding="utf-8", xml_declaration=True)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["地域", "id", "lat", "lon", "polygon_uu",
                      "遊休農地フラグ", "area_m2", "住所", "ラベル", "確信度", "メモ"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n  KML → {kml_path}")
    print(f"  CSV → {csv_path}")
    print("\n  手順:")
    print("    1. KMLをGoogle Earthで開く")
    print("    2. 2023年10月の衛星写真で各ピンを目視確認")
    print("    3. CSVの「ラベル」列に 耕作中 / 放棄地 / 除外 を入力")
    print("    4. combined_labels.csv に追記")


def main(n_samples: int = 150):
    print("香取市・十日町市 ラベリング用KML/CSV生成")
    for region_cfg in REGIONS:
        generate_kml_csv(region_cfg, n_samples)
    print("\n完了")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    main(n)
