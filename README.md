# 衛星リモセンを用いた耕作放棄地検出

Sentinel-1/2衛星データによる2段階の農地スクリーニング研究（茨城県・千葉県）

**農業情報学会 2026年5月発表予定**

---

## 研究概要

農地パトロールの効率化を目的に、衛星データだけで耕作放棄地を自動検出する2段階パイプラインを構築。

```
全農地（筆ポリゴン）
    ↓ 【第1段階】K-Means 教師なしクラスタリング
非水田農地（畑・放棄地候補）
    ↓ 【第2段階】教師あり機械学習
放棄地スコアマップ
```

---

## 第1段階：水田スクリーニング（教師なし）

### 手法

- **アルゴリズム**: K-Means（k=4、エルボー法で決定）
- **水田クラスタ判定**: VH_minが最小のクラスタを水田クラスとして自動同定

### 特徴量（6つ）

| 特徴量 | 定義 | 物理的意味 |
|--------|------|-----------|
| VH_min | 年間VH最小値（湛水期） | 水面鏡面反射によるVH急低下 |
| NDVI_grow | 7〜9月NDVI中央値 | 水稲生育最盛期の植生 |
| VH_winter | 12〜2月VH中央値 | 冬期基準後方散乱 |
| NDVI_flood | 4〜5月NDVI中央値 | 代かき・湛水中の低植生 |
| dNDVI | NDVI_grow − NDVI_flood | NDVI季節振幅 |
| dVH | VH_min − VH_winter | VH季節変化量 |

### 精度結果

| 地域 | 筆数 | Accuracy | IoU |
|------|------|----------|-----|
| つくばみらい市（平野） | 17,420 | 92.78% | 85.80% |
| 稲敷市（平野） | 43,095 | 92.89% | 88.59% |
| 笠間市（丘陵） | 37,238 | 90.81% | 72.91% |

---

## 第2段階：放棄地分類（教師あり）

### 手法

- **モデル**: RF / GBM / SVM / LR / LightGBM / Ensemble
- **特徴量（7つ）**: NDVI_apr, NDVI_flood, dNDVI, dVH, NDWI_mean, VH_min, NDVI_mean
- **評価**: Leave-One-Region-Out 空間交差検証（4地域）

### 教師データ

目視ラベリングによる正解データ（Google Earth で確認）

| 地域 | 放棄地 | 耕作中 | 計 |
|------|--------|--------|-----|
| つくばみらい市 | 95 | 322 | 417 |
| 稲敷市 | 153 | 173 | 326 |
| 笠間市 | 232 | 276 | 508 |
| 香取市 | 58 | 204 | 262 |
| **合計** | **538** | **975** | **1,513** |

### 精度結果（Leave-One-Region-Out・2値分類）

| モデル | F1（放棄地） | IoU |
|--------|------------|-----|
| **SVM** | **77.90%** | **63.80%** |
| LR | 76.95% | 62.54% |
| RF | 76.57% | 62.03% |
| GBM | 75.69% | 60.89% |

### アブレーション実験結果

| 比較 | F1差 | 結論 |
|------|------|------|
| 2段階 vs 1段階（直接2値分類） | +2.14% | 水田除外が有効 |
| area_m2あり vs なし | ≈0% | 地域汎化に寄与しないため除外 |
| v4特徴量追加（DEM・BSI・月次NDVI） | −3% | 過剰追加は汎化を悪化 |

---

## ファイル構成

```
.
├── accuracy_check.py          # 第1段階 精度評価（3地域）
├── abandoned_classifier.py    # 第2段階 放棄地分類・leave-one-out CV・SHAP分析
│
├── data/                      # 特徴量CSV（第1段階 GEEエクスポート）
├── labels/                    # 目視ラベルデータ（combined_labels.csv）
├── figures/                   # 可視化出力（SHAP・精度グラフ）
├── docs/                      # GEEスクリプト
│   ├── gee_features_v4_export.js      # v4特徴量エクスポート（5地域対応）
│   ├── gee_timeseries_export.js       # v2特徴量エクスポート
│   └── gee_base_features_export.js    # 基本特徴量エクスポート
└── scripts/                   # ラベリング補助スクリプト
    ├── create_labeling_kml.py
    ├── create_abandoned_candidate_kml_katori.py
    ├── add_labels_to_combined.py
    └── compare_emaff_baseline.py
```

## 環境

```
Python 3.x
scikit-learn, lightgbm, shap
pandas, numpy, geopandas
matplotlib
```
