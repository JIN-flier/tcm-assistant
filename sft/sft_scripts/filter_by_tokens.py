import argparse
import json
from pathlib import Path

import orjson
from tqdm import tqdm
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    kept = 0
    removed = 0

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f_in, output_path.open(
        "wb",
    ) as f_out:

        for line in tqdm(f_in):
            line = line.strip()

            if not line:
                continue

            total += 1

            row = json.loads(line)

            token_ids = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=False,
            )

            length = len(token_ids)

            if length > args.max_tokens:
                removed += 1
                continue

            f_out.write(
                orjson.dumps(row) + b"\n"
            )

            kept += 1

    print("\n========== Result ==========")
    print(f"Total:   {total}")
    print(f"Kept:    {kept}")
    print(f"Removed: {removed}")
    print(f"Output:  {output_path}")


if __name__ == "__main__":
    main()

