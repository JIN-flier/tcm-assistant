#!/usr/bin/env python3
import argparse
import hashlib
import sqlite3
from pathlib import Path
from tqdm import tqdm


def digest(line: bytes) -> bytes:
    return hashlib.blake2b(line.strip(), digest_size=16).digest()


def create_db(path: Path):
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE IF NOT EXISTS seen (hash BLOB PRIMARY KEY) WITHOUT ROWID')
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=OFF')
    return conn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--database', required=True)
    args = p.parse_args()

    reference, input_path = Path(args.reference), Path(args.input)
    output_path, db_path = Path(args.output), Path(args.database)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = create_db(db_path); cursor = conn.cursor()

    print('Loading domain hashes...')
    reference_count = 0
    with reference.open('rb') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            cursor.execute('INSERT OR IGNORE INTO seen(hash) VALUES (?)', (digest(line),))
            reference_count += 1
            if reference_count % 50000 == 0:
                conn.commit()
    conn.commit()

    print('\nFiltering general dataset...')
    total = kept = duplicate = 0
    with input_path.open('rb') as f_in, output_path.open('wb') as f_out:
        for line in tqdm(f_in):
            line = line.strip()
            if not line:
                continue
            total += 1
            cursor.execute('INSERT OR IGNORE INTO seen(hash) VALUES (?)', (digest(line),))
            if cursor.rowcount == 1:
                f_out.write(line + b'\n'); kept += 1
            else:
                duplicate += 1
            if total % 50000 == 0:
                conn.commit()
    conn.commit(); conn.close()

    print('\n========== Result ==========')
    print(f'Reference domain: {reference_count}')
    print(f'General total:    {total}')
    print(f'General kept:     {kept}')
    print(f'Duplicates:       {duplicate}')
    print(f'Output:           {output_path}')


if __name__ == '__main__':
    main()
