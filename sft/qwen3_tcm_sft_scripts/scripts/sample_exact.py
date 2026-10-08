#!/usr/bin/env python3
import argparse
import hashlib
import sqlite3
from pathlib import Path
from tqdm import tqdm


def stable_hash(line: bytes, seed: int) -> bytes:
    return hashlib.blake2b(str(seed).encode('utf-8') + b':' + line.strip(), digest_size=16).digest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--database', required=True)
    args = p.parse_args()

    input_path, output_path, db_path = Path(args.input), Path(args.output), Path(args.database)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE samples (hash BLOB PRIMARY KEY, line BLOB NOT NULL) WITHOUT ROWID')
    conn.execute('PRAGMA synchronous=OFF'); conn.execute('PRAGMA journal_mode=WAL')
    cursor = conn.cursor(); total = 0

    print('Indexing samples...')
    with input_path.open('rb') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            cursor.execute('INSERT OR IGNORE INTO samples(hash, line) VALUES (?, ?)',
                           (stable_hash(line, args.seed), line))
            total += 1
            if total % 50000 == 0:
                conn.commit()
    conn.commit()

    target = min(args.count, total)
    print(f'Available: {total}')
    print(f'Selecting: {target}')
    with output_path.open('wb') as out:
        cursor.execute('SELECT line FROM samples ORDER BY hash LIMIT ?', (target,))
        while True:
            rows = cursor.fetchmany(10000)
            if not rows:
                break
            for (line,) in rows:
                out.write(line + b'\n')
    conn.close()
    print(f'Output: {output_path}')


if __name__ == '__main__':
    main()
