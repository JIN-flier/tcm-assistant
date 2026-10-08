#!/usr/bin/env python3
import argparse
import hashlib
import sqlite3
from pathlib import Path
from tqdm import tqdm


def create_database(path: Path):
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE IF NOT EXISTS seen (hash BLOB PRIMARY KEY) WITHOUT ROWID')
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=OFF')
    conn.execute('PRAGMA temp_store=MEMORY')
    return conn


def hash_line(line: bytes) -> bytes:
    return hashlib.blake2b(line.strip(), digest_size=16).digest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--database', required=True)
    args = p.parse_args()

    output_path = Path(args.output)
    db_path = Path(args.database)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = create_database(db_path)
    cursor = conn.cursor()
    total = kept = duplicates = 0

    with output_path.open('wb') as out:
        for file_name in args.inputs:
            path = Path(file_name)
            print(f'\nDeduplicating: {path}')
            with path.open('rb') as f:
                for line in tqdm(f):
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    cursor.execute('INSERT OR IGNORE INTO seen(hash) VALUES (?)', (hash_line(line),))
                    if cursor.rowcount == 1:
                        out.write(line + b'\n')
                        kept += 1
                    else:
                        duplicates += 1
                    if total % 50000 == 0:
                        conn.commit()
    conn.commit(); conn.close()
    print('\n========== Dedup Result ==========')
    print(f'Total:      {total}')
    print(f'Kept:       {kept}')
    print(f'Duplicates: {duplicates}')
    print(f'Output:     {output_path}')


if __name__ == '__main__':
    main()
