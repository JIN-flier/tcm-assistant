import argparse
import hashlib
import json
import re
from pathlib import Path

import ijson
import orjson
from tqdm import tqdm


THINK_PATTERN = re.compile(
    r"<think>.*?</think>\s*",
    flags=re.IGNORECASE | re.DOTALL,
)


def first_non_whitespace(path: Path) -> str:
    with path.open("rb") as f:
        while True:
            b = f.read(1)

            if not b:
                return ""

            c = b.decode("utf-8", errors="ignore")

            if not c.isspace():
                return c


def iter_records(path: Path):
    first = first_non_whitespace(path)

    if first == "[":
        with path.open("rb") as f:
            yield from ijson.items(f, "item")

    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line:
                    yield json.loads(line)


def clean_text(value) -> str:
    if value is None:
        return ""

    text = str(value)

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 最多保留两个连续空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_chain_of_thought(text: str) -> str:
    """
    删除显式 <think>...</think>，
    保留最终回答。
    """
    return THINK_PATTERN.sub("", text).strip()


def build_user_content(record: dict) -> str:
    instruction = clean_text(record.get("instruction"))
    input_text = clean_text(record.get("input"))

    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"

    if instruction:
        return instruction

    return input_text


def stable_probability(text: str, seed: int) -> float:
    """
    用 hash 实现确定性采样。
    同一个 seed 下，每次运行结果一致。
    """
    value = f"{seed}:{text}".encode("utf-8")

    digest = hashlib.blake2b(
        value,
        digest_size=8,
    ).digest()

    integer = int.from_bytes(digest, "big")

    return integer / (2**64 - 1)


def convert_record(record: dict):
    user = build_user_content(record)
    assistant = clean_text(record.get("output"))

    # 明确删除显式思维链
    assistant = remove_chain_of_thought(assistant)

    if not user:
        return None

    if not assistant:
        return None

    return {
        "messages": [
            {
                "role": "user",
                "content": user,
            },
            {
                "role": "assistant",
                "content": assistant,
            },
        ]
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    invalid = 0
    sampled_out = 0

    with output_path.open("wb") as out:

        for file_name in args.inputs:
            path = Path(file_name)

            print(f"\nProcessing: {path}")

            for record in tqdm(iter_records(path)):
                total += 1

                converted = convert_record(record)

                if converted is None:
                    invalid += 1
                    continue

                # 根据 user+assistant 做稳定抽样
                sample_key = (
                    converted["messages"][0]["content"]
                    + "\n"
                    + converted["messages"][1]["content"]
                )

                p = stable_probability(
                    sample_key,
                    args.seed,
                )

                if p >= args.sample_rate:
                    sampled_out += 1
                    continue

                out.write(
                    orjson.dumps(converted)
                    + b"\n"
                )

                written += 1

    print("\n========== Done ==========")
    print(f"Total:       {total}")
    print(f"Written:     {written}")
    print(f"Invalid:     {invalid}")
    print(f"Sampled out: {sampled_out}")
    print(f"Output:      {output_path}")


if __name__ == "__main__":
    main()

