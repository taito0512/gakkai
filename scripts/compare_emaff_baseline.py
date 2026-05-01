"""
eMAFF遊休農地フラグ vs 本研究モデル 精度比較

eMAFFフラグを「ベースライン予測器」として扱い、
目視ラベル（ground truth）に対するF1/IoU/Precision/Recallを計算。
本研究のleave-one-out結果と並べて比較する。

eMAFFフラグの定義:
  遊休農地（不耕作緑）/ 遊休農地（不耕作黄）/ 遊休農地（低利用） → 放棄地予測 (1)
  遊休農地ではない → 耕作中予測 (0)
  調査中 / その他 / NaN → 評価対象外（除外）

使い方:
  python scripts/compare_emaff_baseline.py
"""

import numpy as np
import pandas as pd

LABELS_CSV = "labels/combined_labels.csv"

# 本研究モデル（leave-one-out）の結果（abandoned_classifier.pyの出力から転記）
# 3値分類・Ensembleモデル
MODEL_RESULTS = {
    "つくばみらい市": {"F1": 0.615, "IoU": None, "Prec": None, "Rec": None},
    "稲敷市":         {"F1": 0.786, "IoU": None, "Prec": None, "Rec": None},
    "笠間市":         {"F1": 0.844, "IoU": None, "Prec": None, "Rec": None},
    "香取市":         {"F1": None,  "IoU": None, "Prec": None, "Rec": None},
}

# 放棄地フラグと判定するeMAFF値
ABANDONED_FLAGS = {
    "遊休農地（不耕作緑）",
    "遊休農地（不耕作黄）",
    "遊休農地（低利用）",
}
ACTIVE_FLAGS = {"遊休農地ではない"}
UNKNOWN_FLAGS = {"調査中", "その他", "立入困難等外因的理由で調査不可"}


def compute_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou   = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"F1": f1, "Prec": prec, "Rec": rec, "IoU": iou,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def main():
    df = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")

    # 放棄地・耕作中のみ対象
    df = df[df["ラベル"].isin(["放棄地", "耕作中"])].copy()
    df["y_true"] = (df["ラベル"] == "放棄地").astype(int)

    # eMAFFフラグを予測値に変換
    def flag_to_pred(flag):
        if pd.isna(flag):
            return np.nan
        if flag in ABANDONED_FLAGS:
            return 1
        if flag in ACTIVE_FLAGS:
            return 0
        return np.nan  # 調査中・その他は除外

    df["y_emaff"] = df["遊休農地フラグ"].apply(flag_to_pred)

    print("=" * 65)
    print("  eMAFF遊休農地フラグ vs 本研究モデル 精度比較")
    print("=" * 65)
    print("\n【eMAFFフラグの評価可能件数】")
    for region in df["地域"].unique():
        r = df[df["地域"] == region]
        valid = r["y_emaff"].notna()
        abd = (r["y_true"] == 1).sum()
        act = (r["y_true"] == 0).sum()
        ev  = valid.sum()
        print(f"  {region}: 放棄地{abd}件・耕作中{act}件 → フラグ評価可能{ev}件"
              f"（除外{(~valid).sum()}件）")

    print()
    print(f"  {'地域':<14}  {'指標':<6}  {'eMAFFフラグ':>10}  {'本研究モデル':>10}  {'差':>7}")
    print("  " + "─" * 55)

    regions = ["つくばみらい市", "笠間市"]  # 稲敷は全件調査中のため除外
    for region in regions:
        r = df[(df["地域"] == region) & df["y_emaff"].notna()].copy()
        if len(r) == 0:
            print(f"  {region}: フラグ評価可能データなし")
            continue

        m = compute_metrics(r["y_true"].values, r["y_emaff"].values)

        model_f1 = MODEL_RESULTS.get(region, {}).get("F1")

        print(f"\n  [{region}]  評価件数: {len(r)}件"
              f"  (放棄地{r['y_true'].sum()}件 / 耕作中{(r['y_true']==0).sum()}件)")
        print(f"  {'':14}  {'F1':>6}  {m['F1']*100:>9.1f}%"
              + (f"  {model_f1*100:>9.1f}%  {(model_f1-m['F1'])*100:>+6.1f}%" if model_f1 else "  (leave-one-out参照)"))
        print(f"  {'':14}  {'Prec':>6}  {m['Prec']*100:>9.1f}%")
        print(f"  {'':14}  {'Recall':>6}  {m['Rec']*100:>9.1f}%")
        print(f"  {'':14}  {'IoU':>6}  {m['IoU']*100:>9.1f}%")
        print(f"  {'':14}  TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}")

    # 稲敷の説明
    print()
    print("  ※ 稲敷市はeMAFFフラグが全件「調査中」のため比較不可")
    print("     → フラグ未整備地域でも衛星モデルは適用可能（優位性）")

    # 笠間の詳細：「遊休農地ではない」なのに放棄地の件数
    print()
    print("【eMAFFフラグの見逃し分析（笠間市）】")
    kasama = df[df["地域"] == "笠間市"]
    fn_cases = kasama[(kasama["遊休農地フラグ"] == "遊休農地ではない") & (kasama["y_true"] == 1)]
    print(f"  「遊休農地ではない」フラグなのに目視で放棄地: {len(fn_cases)}件")
    print(f"  → eMAFFが見逃した放棄地（偽陰性）")

    fp_cases = kasama[kasama["y_emaff"] == 1]
    tp_cases = kasama[(kasama["y_emaff"] == 1) & (kasama["y_true"] == 1)]
    print(f"  遊休農地フラグありの件数: {len(fp_cases)}件  うち実際に放棄地: {len(tp_cases)}件"
          f"  Precision={len(tp_cases)/len(fp_cases)*100:.1f}%" if len(fp_cases) > 0 else "")


if __name__ == "__main__":
    main()
