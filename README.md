# 水田スクリーニング研究

Sentinel-1/2衛星データを用いた教師なし水田スクリーニング（茨城県3地域）

## 概要

- **手法**: K-Means（k=4）クラスタリング（教師なし）
- **データ**: Sentinel-1（SAR）＋ Sentinel-2（光学）
- **対象地域**: 笠間市・稲敷市・つくばみらい市（茨城県）

## 特徴量

| 特徴量 | 説明 |
|--------|------|
| VH_min | 冬期最小後方散乱（湛水検知） |
| NDVI_grow | 生育期NDVI |
| VH_winter | 冬期後方散乱 |
| NDVI_flood | 湛水期NDVI |
| dNDVI | NDVI_grow − NDVI_flood（季節振幅） |
| dVH | VH_min − VH_winter |

## 精度結果

統一手法（K-Means k=4・6特徴量）による精度：

| 地域 | 筆数 | Accuracy | IoU |
|------|------|----------|-----|
| つくばみらい市 | 17,420 | 92.78% | 85.80% |
| 稲敷市 | 43,095 | 92.89% | 88.59% |
| 笠間市 | 37,238 | 90.81% | 72.91% |

## スクリプト一覧

| ファイル | 内容 |
|----------|------|
| `analyze_rice_paddy.py` | 基本分類（K-Means k=4）、Accuracy/IoU出力 |
| `accuracy_check.py` | 3地域まとめて精度測定（最新手法） |
| `scatter_rice_paddy.py` | 散布図（3地域、クラスタ色分け） |
| `kasama_improvement.py` | 笠間市 精度改善実験 |
| `all_regions_improvement.py` | 3地域 精度改善実験まとめ |
| `multi_agent_discussion.py` | マルチエージェント議論（Anthropic API） |

## 環境

```
Python 3.x
scikit-learn, pandas, numpy, matplotlib
```
