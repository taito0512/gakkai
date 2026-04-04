"""
マルチエージェント議論スクリプト
テーマ: 笠間市のIoU70%をどう改善するか、または現状の精度で実務にどう活用するか
"""

import os
import anthropic

# --- 設定 ---
MODEL = "claude-sonnet-4-20250514"
ROUNDS = 5
MAX_TOKENS = 1024

RESEARCH_CONTEXT = """
【研究概要】
- 手法: Sentinel-1/2衛星データを用いた教師なし水田スクリーニング（K-Means, k=5）
- 特徴量: VH_min（冬期最小後方散乱）, NDVI_grow（生育期NDVI）, VH_winter（冬期後方散乱）, NDVI_flood（湛水期NDVI）
- 検証結果:
    - つくばみらい市: Accuracy 90.2%, IoU 80.1%
    - 稲敷市:         Accuracy 93.3%, IoU 89.3%
    - 笠間市:         Accuracy 90.0%, IoU 70.1%（丘陵地帯、精度低下）
- 課題: 笠間市は丘陵部が多く、地形指標（傾斜・標高）を追加しても精度改善なし
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

DISCUSSION_TOPIC = "笠間市のIoU70%をどう改善するか、または現状の精度で実務にどう活用するか"


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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("環境変数 ANTHROPIC_API_KEY が設定されていません。")

    client = anthropic.Anthropic(api_key=api_key)

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
