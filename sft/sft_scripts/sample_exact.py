import argparse
import hashlib
import sqlite3
from pathlib import Path

from tqdm import tqdm


def stable_hash(line: bytes, seed: int) -> bytes:
    return hashlib.blake2b(
        str(seed).encode("utf-8")
        + b":"
        + line.strip(),
        digest_size=16,
    ).digest()


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
        "--count",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--database",
        required=True,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    db_path = Path(args.database)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE samples (
            hash BLOB PRIMARY KEY,
            line BLOB NOT NULL
        ) WITHOUT ROWID
    """)

    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA journal_mode=WAL")

    cursor = conn.cursor()

    total = 0

    print("Indexing samples...")

    with input_path.open("rb") as f:
        for line in tqdm(f):
            line = line.strip()

            if not line:
                continue

            cursor.execute(
                """
                INSERT OR IGNORE
                INTO samples(hash, line)
                VALUES (?, ?)
                """,
                (
                    stable_hash(line, args.seed),
                    line,
                ),
            )

            total += 1

            if total % 50000 == 0:
                conn.commit()

    conn.commit()

    target = min(
        args.count,
        total,
    )

    print(f"Available: {total}")
    print(f"Selecting: {target}")

    with output_path.open("wb") as out:

        cursor.execute(
            """
            SELECT line
            FROM samples
            ORDER BY hash
            LIMIT ?
            """,
            (target,),
        )

        while True:
            rows = cursor.fetchmany(10000)

            if not rows:
                break

            for (line,) in rows:
                out.write(line + b"\n")

    conn.close()

    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
