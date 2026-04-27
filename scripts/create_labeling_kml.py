"""
教師データ作成用KMLテンプレート生成スクリプト
- eMAFF農地ナビのgeojsonを読み込み、畑のみフィルタしてKMLを生成
- 同時にラベル記録用CSVテンプレートも生成
"""

import json
import sys
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


def create_labeling_kml(geojson_path, region_name="対象地域",
                        kml_output=None, csv_output=None):

    with open(geojson_path, encoding="utf-8-sig") as f:
        data = json.load(f)

    # 畑のみフィルタ
    features_all = data["features"]
    features = [
        f for f in features_all
        if f["properties"].get("ClassificationOfLandCodeName") == "畑"
    ]
    print(f"全件数: {len(features_all)} → 畑のみ: {len(features)} 件")

    # デフォルト出力パス
    if kml_output is None:
        kml_output = f"{region_name}_labeling.kml"
    if csv_output is None:
        csv_output = f"{region_name}_labels.csv"

    # ---- KML生成 ----
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = f"教師データ作成 - {region_name}"

    # スタイル（黄ピン）
    style = ET.SubElement(doc, "Style", id="pin")
    icon_style = ET.SubElement(style, "IconStyle")
    ET.SubElement(icon_style, "scale").text = "1.0"
    icon = ET.SubElement(icon_style, "Icon")
    ET.SubElement(icon, "href").text = \
        "http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png"

    folder = ET.SubElement(doc, "Folder")
    ET.SubElement(folder, "name").text = "畑"

    # CSV用データも同時に準備
    csv_rows = []

    for i, feature in enumerate(features, start=1):
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        lon, lat = coords[0], coords[1]

        usage = props.get("UsageSituationInvestigationResultCodeName", "不明")
        address = props.get("Address", "")
        area = props.get("AreaOnRegistry", "不明")
        pin_id = f"{i:04d}"

        # KMLピン
        pm = ET.SubElement(folder, "Placemark")
        ET.SubElement(pm, "name").text = pin_id
        ET.SubElement(pm, "description").text = (
            f"遊休農地: {usage}\n"
            f"面積: {area}㎡\n"
            f"住所: {address}\n"
            f"lat: {lat:.6f}, lon: {lon:.6f}"
        )
        ET.SubElement(pm, "styleUrl").text = "#pin"
        point = ET.SubElement(pm, "Point")
        ET.SubElement(point, "coordinates").text = f"{lon},{lat},0"

        # CSV行
        csv_rows.append({
            "id": pin_id,
            "lat": f"{lat:.6f}",
            "lon": f"{lon:.6f}",
            "遊休農地フラグ": usage,
            "面積m2": area,
            "住所": address,
            "ラベル": "",        # ← ここに入力: 耕作中 / 放棄地 / 除外
            "確信度": "",        # ← 高 / 中 / 低
            "メモ": "",
        })

    # KML出力
    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    tree.write(kml_output, encoding="utf-8", xml_declaration=True)

    # CSV出力
    with open(csv_output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"KML: {kml_output}")
    print(f"CSV: {csv_output}")
    print(f"\n手順:")
    print(f"  1. {kml_output} をGoogle Earthで開く")
    print(f"  2. 黄色ピンをクリックして場所を確認")
    print(f"  3. {csv_output} をExcel/Numbersで開く")
    print(f"  4. 「ラベル」列に 耕作中 / 放棄地 / 除外 を入力していく")
    print(f"  5. 「確信度」列に 高 / 中 / 低 を入力")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python create_labeling_kml.py <geojson> [地域名]")
        sys.exit(1)
    geojson = sys.argv[1]
    region = sys.argv[2] if len(sys.argv) > 2 else "対象地域"
    create_labeling_kml(
        geojson,
        region,
        kml_output=f"/Users/nagatomotaito/Downloads/{region}_labeling.kml",
        csv_output=f"/Users/nagatomotaito/Downloads/{region}_labels.csv",
    )
