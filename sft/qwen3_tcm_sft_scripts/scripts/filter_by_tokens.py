#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import orjson
from tqdm import tqdm
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--max-tokens', type=int, default=1000)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_path, output_path = Path(args.input), Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = kept = removed = 0

    with input_path.open('r', encoding='utf-8') as f_in, output_path.open('wb') as f_out:
        for line in tqdm(f_in):
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            token_ids = tokenizer.apply_chat_template(
                row['messages'], tokenize=True, add_generation_prompt=False
            )
            if len(token_ids) > args.max_tokens:
                removed += 1
                continue
            f_out.write(orjson.dumps(row) + b'\n')
            kept += 1

    print('\n========== Result ==========')
    print(f'Total:   {total}')
    print(f'Kept:    {kept}')
    print(f'Removed: {removed}')
    print(f'Output:  {output_path}')


if __name__ == '__main__':
    main()
