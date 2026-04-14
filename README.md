# 水田スクリーニング研究

Sentinel-1/2衛星データを用いた教師なし水田スクリーニング（茨城県3地域）

## 概要

- **手法**: K-Means（k=4、エルボー法）クラスタリング（教師なし）
- **データ**: Sentinel-1（SAR）＋ Sentinel-2（光学）
- **対象地域**: 笠間市・稲敷市・つくばみらい市（茨城県）

## 特徴量（6つ）

| 特徴量 | 定義 | 対象期 | 物理的意味 |
|--------|------|--------|-----------|
| VH_min | VH最小値 | DOY 121–166（湛水期） | 水面鏡面反射によるVH急低下を捉える |
| NDVI_grow | NDVI平均 | DOY 167–227（生育最盛期） | 水稲の旺盛な植生を検出する |
| VH_winter | VH平均 | DOY 1–90（冬季非作付期） | 裸地・残渣の散乱特性を把握する |
| NDVI_flood | NDVI平均 | DOY 121–166（湛水期） | 湛水中の低植生状態を確認する |
| dNDVI | NDVI_grow − NDVI_flood | — | NDVI季節振幅により水稲特有の変動を強調する |
| dVH | VH_min − VH_winter | — | VH季節変化量により湛水の後方散乱低下を強調する |

## 水田クラスタ判定

VH_minが最小のクラスタを水田クラスとして自動同定。教師データ・パラメータ調整不要で未知地域へ即時適用が可能。

## 精度結果

統一手法（K-Means k=4・6特徴量）による精度：

| 地域 | 筆数 | Accuracy | IoU |
|------|------|----------|-----|
| つくばみらい市（平野） | 17,420 | 92.78% | 85.80% |
| 稲敷市（平野） | 43,095 | 92.89% | 88.59% |
| 笠間市（丘陵） | 37,238 | 90.81% | 72.91% |

## スクリプト一覧

| ファイル | 内容 |
|----------|------|
| `accuracy_check.py` | 3地域まとめて精度測定（最新手法・6特徴量・k=4） |
| `analyze_rice_paddy.py` | 基本分類（K-Means k=4）、Accuracy/IoU出力 |
| `scatter_rice_paddy.py` | 散布図（3地域、クラスタ色分け） |
| `kasama_improvement.py` | 笠間市 精度改善実験 |
| `all_regions_improvement.py` | 3地域 精度改善実験まとめ |
| `multi_agent_discussion.py` | マルチエージェント議論（Anthropic API） |
| `app.py` | Streamlit Webアプリ（水田判別UI） |

## 環境

```
Python 3.x
scikit-learn, pandas, numpy, matplotlib
```
