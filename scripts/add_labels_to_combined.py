"""
ラベリング済みCSVをcombined_labels.csvに追記するスクリプト

使い方:
  python scripts/add_labels_to_combined.py ~/Downloads/香取市_labeling_v2_150abd_80act_labels.csv

処理:
  1. 入力CSVを読み込み（ラベル列が空欄の行は除外）
  2. combined_labels.csvの列形式に変換
  3. 重複チェック（lat/lonが既存と近い行はスキップ）
  4. 追記してバックアップ保存
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

COMBINED_CSV  = Path("labels/combined_labels.csv")
MATCH_THR     = 0.001   # 度（約100m）以内は重複とみなす


def main(input_path: str):
    inp = Path(input_path)
    if not inp.exists():
        print(f"[ERROR] ファイルが見つかりません: {inp}")
        sys.exit(1)

    # ---- 入力CSV読み込み ----
    new_df = pd.read_csv(inp, encoding="utf-8-sig")
    print(f"入力CSV: {len(new_df)}行")

    # ラベル未入力行を除外
    new_df = new_df[new_df["ラベル"].notna() & (new_df["ラベル"].astype(str).str.strip() != "")].copy()
    print(f"ラベル入力済み: {len(new_df)}行")
    if len(new_df) == 0:
        print("追記対象なし。終了。")
        return

    # ---- combined_labels.csv 読み込み ----
    combined = pd.read_csv(COMBINED_CSV, encoding="utf-8-sig")
    print(f"combined_labels.csv: {len(combined)}行（追記前）")

    # ---- 重複チェック ----
    existing_coords = combined[["lat", "lon"]].dropna().values
    new_coords = new_df[["lat", "lon"]].values.astype(float)
    if len(existing_coords) > 0:
        tree = cKDTree(existing_coords)
        dists, _ = tree.query(new_coords, k=1)
        duplicates = dists < MATCH_THR
        if duplicates.sum() > 0:
            print(f"重複スキップ: {duplicates.sum()}件（既に combined_labels.csv に含まれる）")
        new_df = new_df[~duplicates].copy()
    print(f"追記対象: {len(new_df)}件")
    if len(new_df) == 0:
        print("全件重複のため追記なし。終了。")
        return

    # ---- combined_labels.csv の列形式に変換 ----
    # 候補種別からサンプル種別を決定
    def to_sample_type(row):
        if "候補種別" in row and pd.notna(row.get("候補種別")):
            if "放棄地" in str(row["候補種別"]):
                return "abandoned_candidate"
            if "耕作中" in str(row["候補種別"]):
                return "farm_candidate"
        return "random"

    rows = []
    for _, row in new_df.iterrows():
        rows.append({
            "地域":       row.get("地域", ""),
            "サンプル種別": to_sample_type(row),
            "id":         row.get("id", ""),
            "lat":        row["lat"],
            "lon":        row["lon"],
            "遊休農地フラグ": row.get("遊休農地フラグ", ""),
            "面積m2":     row.get("面積m2", ""),
            "住所":       row.get("住所", ""),
            "ラベル":     str(row["ラベル"]).strip(),
            "確信度":     row.get("確信度", ""),
            "メモ":       row.get("メモ", ""),
            "画像日付":   row.get("画像日付", ""),
            "参照情報":   row.get("参照情報", ""),
            "判読者":     "Claude",
            "判読回":     "追加",
        })

    append_df = pd.DataFrame(rows)

    # ---- バックアップ ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = COMBINED_CSV.with_name(f"combined_labels_backup_{ts}.csv")
    shutil.copy(COMBINED_CSV, backup_path)
    print(f"バックアップ: {backup_path}")

    # ---- 追記 ----
    updated = pd.concat([combined, append_df], ignore_index=True)
    updated.to_csv(COMBINED_CSV, index=False, encoding="utf-8-sig")
    print(f"combined_labels.csv: {len(combined)}行 → {len(updated)}行（+{len(append_df)}件追記）")

    # ---- 追記結果のサマリー ----
    print("\n【追記内容】")
    print(append_df.groupby(["地域", "ラベル"]).size().to_string())
    print("\n完了！")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python scripts/add_labels_to_combined.py <ラベリング済みCSVパス>")
        sys.exit(1)
    main(sys.argv[1])
