"""
放棄地が多そうな追加候補KML/CSVを作成する。

優先度は以下でスコア化する。
- eMAFFの利用状況が「立入困難」「調査中」「その他」
- 既に目視で放棄地と判定した点に近い
- 面積が大きい

既にラベル済みの点は除外し、スコア順に上位N件を出力する。
"""

import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


CSV_COLUMNS = [
    "id",
    "lat",
    "lon",
    "遊休農地フラグ",
    "面積m2",
    "住所",
    "候補スコア",
    "候補理由",
    "ラベル",
    "確信度",
    "メモ",
    "画像日付",
    "参照情報",
    "判読者",
    "判読回",
]


def load_geojson(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def load_labeled_points(path):
    labeled_keys = set()
    abandoned_points = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lat = float(row["lat"])
            lon = float(row["lon"])
            labeled_keys.add((round(lat, 6), round(lon, 6)))
            if row.get("ラベル") == "放棄地":
                abandoned_points.append((lat, lon))
    return labeled_keys, abandoned_points


def get_point(feature):
    coords = feature["geometry"]["coordinates"]
    if feature["geometry"]["type"] != "Point":
        raise ValueError(f"Point geometry expected: {feature['geometry']['type']}")
    return float(coords[1]), float(coords[0])


def distance_m(lat1, lon1, lat2, lon2):
    radius = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def nearest_distance(lat, lon, points):
    if not points:
        return None
    return min(distance_m(lat, lon, p_lat, p_lon) for p_lat, p_lon in points)


def score_feature(feature, abandoned_points):
    props = feature["properties"]
    usage = props.get("UsageSituationInvestigationResultCodeName", "")
    area = float(props.get("AreaOnRegistry") or 0)
    lat, lon = get_point(feature)
    reasons = []
    score = 0.0

    usage_scores = {
        "立入困難等外因的理由で調査不可": 90,
        "調査中": 70,
        "その他": 60,
        "遊休農地ではない": 0,
    }
    usage_score = usage_scores.get(usage, 0)
    if usage_score:
        score += usage_score
        reasons.append(f"eMAFF:{usage}")

    dist = nearest_distance(lat, lon, abandoned_points)
    if dist is not None:
        if dist <= 150:
            score += 55
            reasons.append("既判定放棄地150m以内")
        elif dist <= 300:
            score += 40
            reasons.append("既判定放棄地300m以内")
        elif dist <= 600:
            score += 25
            reasons.append("既判定放棄地600m以内")
        elif dist <= 1000:
            score += 10
            reasons.append("既判定放棄地1km以内")

    if area >= 3000:
        score += 20
        reasons.append("大面積3000m2以上")
    elif area >= 1500:
        score += 12
        reasons.append("中大面積1500m2以上")
    elif area >= 800:
        score += 6
        reasons.append("面積800m2以上")

    # 極小区画は目視でも除外になりやすいので少し下げる。
    if area < 100:
        score -= 15
        reasons.append("極小区画")

    return score, "; ".join(reasons) if reasons else "近傍候補"


def style_for_rank(rank, limit):
    third = limit // 3
    if rank <= third:
        return "high"
    if rank <= third * 2:
        return "medium"
    return "low"


def add_style(doc, style_id, href):
    style = ET.SubElement(doc, "Style", id=style_id)
    icon_style = ET.SubElement(style, "IconStyle")
    ET.SubElement(icon_style, "scale").text = "1.0"
    icon = ET.SubElement(icon_style, "Icon")
    ET.SubElement(icon, "href").text = href


def create_outputs(geojson_path, labeled_csv, output_dir, limit=750, region=None, land_types=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 地域名を自動推定（引数がなければCSVファイル名から）
    if region is None:
        region = Path(labeled_csv).stem.replace("_labels", "")

    # 対象地目（デフォルトは畑のみ）
    if land_types is None:
        land_types = ["畑"]

    data = load_geojson(geojson_path)
    labeled_keys, abandoned_points = load_labeled_points(labeled_csv)

    candidates = []
    for feature in data["features"]:
        props = feature["properties"]
        if props.get("ClassificationOfLandCodeName") not in land_types:
            continue
        lat, lon = get_point(feature)
        if (round(lat, 6), round(lon, 6)) in labeled_keys:
            continue
        score, reason = score_feature(feature, abandoned_points)
        candidates.append((score, reason, feature))

    candidates.sort(
        key=lambda item: (
            -item[0],
            -float(item[2]["properties"].get("AreaOnRegistry") or 0),
        )
    )
    candidates = candidates[:limit]

    kml_path = output_dir / f"{region}_likely_abandoned_next{limit}.kml"
    csv_path = output_dir / f"{region}_likely_abandoned_next{limit}_labels.csv"

    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = f"{region} 放棄地追加候補 next{limit}"
    add_style(doc, "high", "http://maps.google.com/mapfiles/kml/paddle/red-circle.png")
    add_style(doc, "medium", "http://maps.google.com/mapfiles/kml/paddle/orange-circle.png")
    add_style(doc, "low", "http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png")

    actual = len(candidates)
    third = actual // 3
    folders = {}
    for name in ("high", "medium", "low"):
        folder = ET.SubElement(doc, "Folder")
        ET.SubElement(folder, "name").text = {
            "high": f"高優先 1-{third}",
            "medium": f"中優先 {third+1}-{third*2}",
            "low": f"低優先 {third*2+1}-{actual}",
        }[name]
        folders[name] = folder

    rows = []
    for rank, (score, reason, feature) in enumerate(candidates, start=1):
        props = feature["properties"]
        lat, lon = get_point(feature)
        pin_id = f"B{rank:04d}"
        usage = props.get("UsageSituationInvestigationResultCodeName", "")
        area = props.get("AreaOnRegistry", "")
        address = props.get("Address", "")
        priority = style_for_rank(rank, actual)

        pm = ET.SubElement(folders[priority], "Placemark")
        ET.SubElement(pm, "name").text = pin_id
        ET.SubElement(pm, "description").text = (
            f"候補スコア: {score:.1f}\n"
            f"候補理由: {reason}\n"
            f"遊休農地フラグ: {usage}\n"
            f"面積: {area}㎡\n"
            f"住所: {address}\n"
            f"lat: {lat:.6f}, lon: {lon:.6f}"
        )
        ET.SubElement(pm, "styleUrl").text = f"#{priority}"
        point = ET.SubElement(pm, "Point")
        ET.SubElement(point, "coordinates").text = f"{lon},{lat},0"

        rows.append({
            "id": pin_id,
            "lat": f"{lat:.6f}",
            "lon": f"{lon:.6f}",
            "遊休農地フラグ": usage,
            "面積m2": area,
            "住所": address,
            "候補スコア": f"{score:.1f}",
            "候補理由": reason,
            "ラベル": "",
            "確信度": "",
            "メモ": "",
            "画像日付": "",
            "参照情報": f"{region}追加候補; 放棄地近傍/eMAFF属性スコア",
            "判読者": "",
            "判読回": "",
        })

    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    tree.write(kml_path, encoding="utf-8", xml_declaration=True)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"candidates={len(rows)}")
    print(f"kml={kml_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("usage: python scripts/create_likely_abandoned_pool.py <geojson> <labeled_csv> <output_dir> <limit> [region] [land_types(カンマ区切り)]")
        sys.exit(1)
    region = sys.argv[5] if len(sys.argv) > 5 else None
    land_types = sys.argv[6].split(",") if len(sys.argv) > 6 else None
    create_outputs(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), region, land_types)
