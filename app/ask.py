"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保持して対話する

ツールは持たないが、--chat モードでは会話履歴を保持する。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import os
import time

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, history: list[types.Content] | None = None) -> str:
    temperature_text = os.getenv("AIE_TEMPERATURE", "1.0")
    try:
        temperature = float(temperature_text)
    except ValueError:
        raise ValueError(f"AIE_TEMPERATUREは数値で指定してください: {temperature_text}")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(f"AIE_TEMPERATUREは0.0以上2.0以下で指定してください: {temperature}")

    contents = list(history or [])
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    client = genai.Client(api_key=get_api_key())
    config = types.GenerateContentConfig(temperature=temperature)
    started_at = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=contents, config=config)
    elapsed_seconds = time.perf_counter() - started_at

    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
    print(f"[応答時間] {elapsed_seconds:.3f}秒")
    print(f"[入力トークン] {input_tokens}")
    print(f"[出力トークン] {output_tokens}")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保持して対話する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[temperature] {os.getenv('AIE_TEMPERATURE', '1.0')}")

    if args.chat:
        history: list[types.Content] = []
        print("対話モードです。終了するには exit または quit を入力してください。")
        try:
            while True:
                question = input("[質問] ").strip()
                if question.lower() in {"exit", "quit"}:
                    break
                if not question:
                    continue
                try:
                    answer = ask(question, history=history)
                except errors.APIError as e:
                    print(api_error_message(e.code, e.message))
                    continue
                print(f"\n[回答]\n{answer}\n")
                history.append(types.Content(role="user", parts=[types.Part(text=question)]))
                history.append(types.Content(role="model", parts=[types.Part(text=answer)]))
        except (EOFError, KeyboardInterrupt):
            print("\n対話を終了します。")
        return

    question = args.question or DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    try:
        print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
