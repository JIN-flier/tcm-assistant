#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path
import orjson
from tqdm import tqdm

EMPTY_THINK = '<think>\n\n</think>\n\n'
THINK_RE = re.compile(r'^\s*<think>.*?</think>\s*', flags=re.DOTALL | re.IGNORECASE)


def probability(data: bytes, seed: int) -> float:
    h = hashlib.blake2b(str(seed).encode('utf-8') + b':' + data, digest_size=8).digest()
    return int.from_bytes(h, byteorder='big', signed=False) / (2**64 - 1)


def prepare_messages(row):
    messages = row.get('messages')
    if not messages:
        return None
    new_messages = []
    for message in messages:
        role = message.get('role')
        content = str(message.get('content', '')).strip()
        if not role or not content:
            return None
        if role == 'assistant':
            content = THINK_RE.sub('', content).strip()
            if not content:
                return None
            content = EMPTY_THINK + content
        new_messages.append({'role': role, 'content': content})
    return {'messages': new_messages}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--val-ratio', type=float, default=0.001)
    p.add_argument('--test-ratio', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / 'train.jsonl'
    val_path = output_dir / 'validation.jsonl'
    test_path = output_dir / 'test.jsonl'
    counts = {'train': 0, 'validation': 0, 'test': 0, 'invalid': 0}

    with train_path.open('wb') as train_out, val_path.open('wb') as val_out, test_path.open('wb') as test_out:
        for file_name in args.inputs:
            path = Path(file_name)
            print(f'\nProcessing {path}')
            with path.open('r', encoding='utf-8') as f:
                for line in tqdm(f):
                    line = line.strip()
                    if not line:
                        continue
                    prepared = prepare_messages(json.loads(line))
                    if prepared is None:
                        counts['invalid'] += 1
                        continue
                    encoded = orjson.dumps(prepared)
                    pval = probability(encoded, args.seed)
                    if pval < args.test_ratio:
                        test_out.write(encoded + b'\n'); counts['test'] += 1
                    elif pval < args.test_ratio + args.val_ratio:
                        val_out.write(encoded + b'\n'); counts['validation'] += 1
                    else:
                        train_out.write(encoded + b'\n'); counts['train'] += 1

    print('\n========== Final Dataset ==========')
    for key, value in counts.items():
        print(f'{key:12}: {value:,}')
    print(f'\nTrain: {train_path}\nVal:   {val_path}\nTest:  {test_path}')


if __name__ == '__main__':
    main()
