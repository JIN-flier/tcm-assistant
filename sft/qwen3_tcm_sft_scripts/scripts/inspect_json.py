#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import ijson


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--n', type=int, default=3)
    args = p.parse_args()
    path = Path(args.path)
    print(f'File: {path}')
    print(f'Size: {path.stat().st_size / 1024 / 1024:.2f} MB')
    for i, record in enumerate(iter_records(path)):
        print(f'\n========== Record {i} ==========')
        print('keys:', list(record.keys()))
        for key, value in record.items():
            text = str(value)
            if len(text) > 500:
                text = text[:500] + '...'
            print(f'{key}: {text}')
        if i + 1 >= args.n:
            break


if __name__ == '__main__':
    main()
