"""
マルチエージェント議論スクリプト
テーマ: 最新論文の知見を踏まえ、笠間市IoU74.8%からの次の一手と教師なし手法の将来性
"""

import os
import anthropic

# --- 設定 ---
MODEL = "claude-sonnet-4-6"
ROUNDS = 5
MAX_TOKENS = 1024

RESEARCH_CONTEXT = """
【研究概要】
Sentinel-1/2衛星データを用いた教師なし水田スクリーニング（茨城県3地域）

■ 確定した統一手法
  特徴量: VH_min, NDVI_grow, VH_winter, NDVI_flood（ベース4）
          + dNDVI(=NDVI_grow-NDVI_flood), dVH(=VH_min-VH_winter)（派生2）= 計6特徴量
  アルゴリズム: K-Means(k=4) 3地域統一

■ 精度結果（統一手法）
  つくばみらい市: IoU 85.80%（平野部）
  稲敷市:         IoU 88.59%（平野部）
  笠間市:         IoU 72.91%（丘陵部）

■ 特徴量寄与率（ablation study, 3地域平均IoU低下）
  VH_winter除外: -14.4%（最重要）
  VH_min除外:    -11.1%
  NDVI_grow除外: -6.0%
  NDVI_flood除外:-3.5%
  dNDVI除外:     -1.6%
  dVH除外:       -0.2%（ほぼ無影響）

■ 参照データ
  JAXA HRLULC（土地利用分類）を正解ラベルとして使用

■ 笠間市が難しい理由
  丘陵地帯が多く、SARの地形散乱（フォアショートニング）が水田検知を阻害。
  水田比率が30%と低く（他2地域は50〜61%）、教師なし手法のクラスタ割当が困難。
  耕作放棄地との誤検知分離が主要課題。

■ 発表予定
  農業情報学会（5月末）、口頭発表15分
"""

AGENTS = [
    {
        "name": "農業リモートセンシング研究者",
        "role": "農業リモートセンシング研究者（衛星データと農業の専門家）",
        "style": "衛星データの特性・SAR解析・農業応用の専門知識を活かして発言する。専門的で論理的。",
    },
    {
        "name": "機械学習エンジニア",
        "role": "機械学習エンジニア（分類手法・特徴量設計の専門家）",
        "style": "モデルの改善案・特徴量エンジニアリング・代替アルゴリズムの観点から発言する。実装寄りで具体的。",
    },
    {
        "name": "農業行政の実務者",
        "role": "農業行政の実務者（農地パトロールの現場を知る人）",
        "style": "現場の実情・行政コスト・農家との関係・実務での使いやすさを重視して発言する。現実的。",
    },
]

MODERATOR = {
    "name": "司会・総括役",
    "role": "議論の司会者・総括役",
    "style": "議論全体を俯瞰し、各専門家の意見を整理して総括する。公平で建設的。",
}

DISCUSSION_TOPIC = """
農業情報学会（5月末、口頭発表15分）に向けて、現在の研究内容に対して
査読者・聴衆から突っ込まれそうな穴・弱点を洗い出してほしい。

現在の状況：
- 統一手法（6特徴量・k=4）で3地域の精度確定済み
- ablation studyでVH_winterが最重要（-14.4%）と判明
- 笠間市で目視確認実施（FP=山林/耕作放棄地、FN=全件水田と確認）
- 旧手法（4特徴量・k=5）と新手法の分類結果が98.2%一致することを確認
- JAXA HRLULCを参照データとして使用（122地点目視で87.5%一致を確認済み）
- Foliumで航空写真上に分類結果を可視化するデモを用意

以下の観点で議論してほしい：
(1) 研究の設計・手法面で査読者に突っ込まれそうな弱点はどこか
(2) 精度評価・検証の面で不十分な点はあるか
(3) 実用性・社会実装の観点で説明が足りていない点はあるか
"""


def build_system_prompt(agent: dict) -> str:
    return f"""あなたは「{agent['role']}」として議論に参加しています。

{RESEARCH_CONTEXT}

議論テーマ: {DISCUSSION_TOPIC}

【発言スタイル】
{agent['style']}

【ルール】
- 必ず日本語で発言する
- 前の発言者の意見を踏まえて、自分の専門的な視点で意見を述べる
- 発言は200〜300字程度で簡潔にまとめる
- 自分の名前（{agent['name']}）を文頭に記載する
"""


def build_moderator_prompt() -> str:
    return f"""あなたは「{MODERATOR['role']}」として、専門家3人（農業リモートセンシング研究者・機械学習エンジニア・農業行政の実務者）の議論を総括します。

{RESEARCH_CONTEXT}

議論テーマ: {DISCUSSION_TOPIC}

【ルール】
- 必ず日本語で発言する
- 各専門家の意見の要点を整理する
- 議論から導き出される結論・提言をまとめる
- 総括は400〜600字程度でまとめる
- 「【議論の総括】」という見出しで始める
"""


def call_agent(client: anthropic.Anthropic, agent: dict, conversation_history: list) -> str:
    """エージェントを呼び出して発言を生成する"""
    system = build_system_prompt(agent)

    # 会話履歴をユーザーメッセージとして渡す
    if conversation_history:
        history_text = "\n".join(conversation_history)
        messages = [
            {"role": "user", "content": f"これまでの議論:\n{history_text}\n\nあなたの番です。発言してください。"}
        ]
    else:
        messages = [
            {"role": "user", "content": "議論を開始します。最初に発言してください。"}
        ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
    )
    return response.content[0].text.strip()


def call_moderator(client: anthropic.Anthropic, conversation_history: list) -> str:
    """司会者が議論を総括する"""
    system = build_moderator_prompt()
    history_text = "\n".join(conversation_history)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS * 2,
        system=system,
        messages=[
            {"role": "user", "content": f"以下の議論全体を総括してください。\n\n{history_text}"}
        ],
    )
    return response.content[0].text.strip()


def main():
    client = anthropic.Anthropic()  # 環境変数から自動取得

    print("=" * 60)
    print("マルチエージェント議論")
    print(f"テーマ: {DISCUSSION_TOPIC}")
    print("=" * 60)
    print()

    conversation_history: list[str] = []

    # --- 5ラウンドの議論 ---
    for round_num in range(1, ROUNDS + 1):
        print(f"【ラウンド {round_num}】")
        print("-" * 40)

        for agent in AGENTS:
            print(f"\n▶ {agent['name']} が発言中...")
            utterance = call_agent(client, agent, conversation_history)
            conversation_history.append(utterance)
            print(utterance)
            print()

    # --- 司会による総括 ---
    print("=" * 60)
    print("【最終総括】")
    print("-" * 40)
    print(f"\n▶ {MODERATOR['name']} が総括中...")
    summary = call_moderator(client, conversation_history)
    print(summary)
    print()
    print("=" * 60)
    print("議論終了")


if __name__ == "__main__":
    main()
