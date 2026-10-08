#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path
import ijson
import orjson
from tqdm import tqdm

THINK_PATTERN = re.compile(r'<think>.*?</think>\s*', flags=re.IGNORECASE | re.DOTALL)


def first_non_whitespace(path: Path) -> str:
    with path.open('rb') as f:
        while True:
            b = f.read(1)
            if not b:
                return ''
            c = b.decode('utf-8', errors='ignore')
            if not c.isspace():
                return c


def iter_records(path: Path):
    first = first_non_whitespace(path)
    if first == '[':
        with path.open('rb') as f:
            yield from ijson.items(f, 'item')
    else:
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def clean_text(value) -> str:
    if value is None:
        return ''
    text = str(value).replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def remove_chain_of_thought(text: str) -> str:
    return THINK_PATTERN.sub('', text).strip()


def build_user_content(record: dict) -> str:
    instruction = clean_text(record.get('instruction'))
    input_text = clean_text(record.get('input'))
    if instruction and input_text:
        return f'{instruction}\n\n{input_text}'
    return instruction or input_text


def stable_probability(text: str, seed: int) -> float:
    digest = hashlib.blake2b(f'{seed}:{text}'.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big') / (2**64 - 1)


def convert_record(record: dict):
    user = build_user_content(record)
    assistant = remove_chain_of_thought(clean_text(record.get('output')))
    if not user or not assistant:
        return None
    return {
        'messages': [
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': assistant},
        ]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--sample-rate', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = written = invalid = sampled_out = 0

    with output_path.open('wb') as out:
        for file_name in args.inputs:
            path = Path(file_name)
            print(f'\nProcessing: {path}')
            for record in tqdm(iter_records(path)):
                total += 1
                converted = convert_record(record)
                if converted is None:
                    invalid += 1
                    continue
                sample_key = converted['messages'][0]['content'] + '\n' + converted['messages'][1]['content']
                if stable_probability(sample_key, args.seed) >= args.sample_rate:
                    sampled_out += 1
                    continue
                out.write(orjson.dumps(converted) + b'\n')
                written += 1

    print('\n========== Done ==========')
    print(f'Total:       {total}')
    print(f'Written:     {written}')
    print(f'Invalid:     {invalid}')
    print(f'Sampled out: {sampled_out}')
    print(f'Output:      {output_path}')


if __name__ == '__main__':
    main()
