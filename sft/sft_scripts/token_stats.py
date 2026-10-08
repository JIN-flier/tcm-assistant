import argparse
import json
from collections import Counter

from tqdm import tqdm
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="本地 Qwen 模型目录",
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    buckets = Counter()

    total = 0
    maximum = 0

    with open(
        args.dataset,
        "r",
        encoding="utf-8",
    ) as f:

        for line in tqdm(f):
            row = json.loads(line)

            ids = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=False,
            )

            n = len(ids)

            total += 1
            maximum = max(maximum, n)

            if n <= 1024:
                buckets["<=1024"] += 1
            elif n <= 2048:
                buckets["1025-2048"] += 1
            elif n <= 4096:
                buckets["2049-4096"] += 1
            elif n <= 8192:
                buckets["4097-8192"] += 1
            elif n <= 16384:
                buckets["8193-16384"] += 1
            else:
                buckets[">16384"] += 1

    print("\n========== Token Statistics ==========")
    print(f"Total: {total}")
    print(f"Maximum tokens: {maximum}")

    for key, value in buckets.items():
        ratio = value / total * 100

        print(
            f"{key:>12}: "
            f"{value:>10} "
            f"({ratio:.2f}%)"
        )


if __name__ == "__main__":
    main()
