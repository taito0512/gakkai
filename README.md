# 水田スクリーニング研究

Sentinel-1/2衛星データを用いた教師なし水田スクリーニング（茨城県3地域）

## 概要

- **手法**: K-Means / GMM クラスタリング（教師なし）
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

| 地域 | ベースライン IoU | 最良 IoU | 改善幅 | 最良手法 |
|------|----------------|---------|--------|---------|
| 笠間市 | 70.1% | 73.9% | +3.8pp | +空間座標 × GMM(k=5) |
| 稲敷市 | 89.3% | 90.3% | +1.0pp | +dNDVI+dVH+空間 × K-Means(k=7) |
| つくばみらい市 | 80.1% | 87.6% | +7.5pp | +dNDVI+dVH × K-Means(k=8) |

## スクリプト一覧

| ファイル | 内容 |
|----------|------|
| `analyze_rice_paddy.py` | 基本分類（K-Means k=5）、Accuracy/IoU出力 |
| `scatter_rice_paddy.py` | 散布図（3地域、クラスタ色分け） |
| `kasama_improvement.py` | 笠間市 精度改善実験 |
| `all_regions_improvement.py` | 3地域 精度改善実験まとめ |
| `multi_agent_discussion.py` | マルチエージェント議論（Anthropic API） |

## 環境

```
Python 3.x
scikit-learn, pandas, numpy, matplotlib
```
