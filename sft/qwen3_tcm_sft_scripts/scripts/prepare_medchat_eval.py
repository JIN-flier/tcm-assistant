#!/usr/bin/env python3
"""Recover the *sampled-out* MedChatZH records using the original SFT hash rule.

Only these candidates are sampled, then checked against ALL actual split files.
Full/question-normalized duplicates and a conservative fuzzy check are removed.
No claims of semantic decontamination are made.
"""
import argparse
import hashlib
import heapq
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


def records(path):
    with path.open('rb') as fh:
        first = b''
        while True:
            b = fh.read(1)
            if not b: return
            if not b.isspace():
                first = b; break
    if first == b'[':
        try: import ijson
        except ImportError as exc: raise SystemExit('JSON array requires ijson: uv pip install ijson') from exc
        with path.open('rb') as fh:
            yield from ijson.items(fh, 'item')
    else:
        with path.open('rb') as fh:
            for line in fh:
                if line.strip(): yield json.loads(line)


def clean(value):
    if value is None: return ''
    value = str(value).replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
    return re.sub(r'\n{3,}', '\n\n', value).strip()


def remove_cot(value):
    return re.sub(r'<think>.*?</think>\s*', '', value, flags=re.IGNORECASE | re.DOTALL).strip()


def convert(row):
    instruction = clean(row.get('instruction'))
    extra = clean(row.get('input'))
    question = (instruction + '\n\n' + extra) if instruction and extra else instruction or extra
    answer = remove_cot(clean(row.get('output')))
    if not question or not answer: return None
    return question, answer


def stable_probability(text, seed):
    # Exact implementation of scripts/normalize_sft.py stable_probability.
    digest = hashlib.blake2b(f'{seed}:{text}'.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big') / (2**64 - 1)


def normalized(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    # Ignore spaces/punctuation for conservative question-only exact matching.
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))


def row_prompt(row):
    msgs = row.get('messages', [])
    return next((str(m.get('content', '')) for m in msgs if m.get('role') == 'user'), '')


def digest(text):
    return hashlib.blake2b(text.encode('utf-8'), digest_size=16).hexdigest()


def candidate_buckets(pool):
    d = defaultdict(set)
    for q in pool:
        if len(q) >= 45:
            d[('prefix', q[:18])].add(q)
            d[('suffix', q[-18:])].add(q)
    return d


def fuzzy_candidates(q, buckets):
    if len(q) < 45: return ()
    return (buckets.get(('prefix', q[:18]), set()) |
            buckets.get(('suffix', q[-18:]), set()))


def prepare(args):
    if not 0 < args.sample_rate < 1: raise SystemExit('--sample-rate must be in (0,1)')
    if args.pool_factor < 1 or args.limit < 1: raise SystemExit('limit/pool-factor must be positive')
    if not all(Path(p).is_file() for p in [args.raw, *args.splits]):
        raise SystemExit('Raw file and all --splits must exist. Do not evaluate without split files.')
    capacity = args.limit * args.pool_factor
    # Select a stable, random-looking top-K by distinct QUESTION (not answer).
    # Store no data from all ~764k inputs in RAM.
    heap, keep = [], {}
    counts = defaultdict(int)
    for line_no, rec in enumerate(records(Path(args.raw)), start=1):
        counts['raw'] += 1
        qa = convert(rec)
        if not qa: counts['invalid'] += 1; continue
        q, answer = qa
        if stable_probability(q + '\n' + answer, args.seed) < args.sample_rate:
            counts['sampled_in_training_pipeline'] += 1; continue
        counts['sampled_out'] += 1
        if not args.min_question_chars <= len(q) <= args.max_question_chars:
            counts['question_length_filtered'] += 1; continue
        if len(answer) < args.min_reference_chars:
            counts['reference_length_filtered'] += 1; continue
        canon = normalized(q)
        if not canon: continue
        h = int(digest(f'{args.seed}:{canon}'), 16)
        if canon in keep:
            counts['candidate_question_duplicates'] += 1
            continue
        obj = (h, canon, q, answer, line_no)
        if len(keep) < capacity:
            keep[canon] = obj
            heapq.heappush(heap, (-h, canon))
        elif h < -heap[0][0]:
            _, stale = heapq.heappop(heap)
            del keep[stale]
            keep[canon] = obj
            heapq.heappush(heap, (-h, canon))
    counts['candidate_pool'] = len(keep)
    if not keep: raise SystemExit('No sampled-out candidates. Check sample-rate and input.')

    buckets = candidate_buckets(keep) if args.fuzzy else {}
    excluded = set()
    # All train/val/test contain original and other-source data, so one scan suffices
    # to check cross-dataset leakage *at the level of a question*.
    for path in args.splits:
        print(f'Checking actual training split for leakage: {path}', flush=True)
        for row in records(Path(path)):
            counts['split_rows_scanned'] += 1
            q = normalized(row_prompt(row))
            if not q: continue
            if q in keep:
                if q not in excluded: counts['exact_question_overlap'] += 1
                excluded.add(q)
            if not args.fuzzy or len(q) < 45: continue
            for candidate in fuzzy_candidates(q, buckets):
                if candidate in excluded: continue
                if abs(len(candidate) - len(q)) > .10 * max(len(candidate), len(q)):
                    continue
                # A conservative near-copy detector, NOT semantic paraphrase detection.
                from rapidfuzz.fuzz import ratio
                if ratio(candidate, q) >= args.fuzzy_threshold * 100:
                    excluded.add(candidate)
                    counts['near_text_overlap'] += 1
        print(f"  scanned={counts['split_rows_scanned']:,} excluded={len(excluded):,}", flush=True)

    eligible = [v for k, v in keep.items() if k not in excluded]
    eligible.sort(key=lambda obj: (obj[0], obj[1]))
    selected = eligible[:args.limit]
    if len(selected) < args.limit:
        print(f'WARNING: only {len(selected)} eligible records; consider increasing --pool-factor.', flush=True)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as fh:
        for _, canon, q, answer, source_row in selected:
            fh.write(json.dumps({
                'id': digest(canon), 'question': q, 'reference': answer,
                'source': 'MedChatZH_train.json', 'original_row': source_row,
            }, ensure_ascii=False) + '\n')
    counts['eligible'] = len(eligible)
    counts['written'] = len(selected)
    counts['seed'] = args.seed
    counts['sample_rate'] = args.sample_rate
    counts['fuzzy'] = args.fuzzy
    counts['fuzzy_threshold'] = args.fuzzy_threshold
    counts['limitation'] = 'No semantic paraphrase decontamination; verify clinical accuracy and dataset overlap independently.'
    report = out.with_suffix('.audit.json')
    report.write_text(json.dumps(dict(counts), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {len(selected)} examples -> {out}; audit -> {report}')
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True, type=Path)
    p.add_argument('--splits', nargs='+', required=True, type=Path, help='All actual train/val/test JSONL files')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--limit', type=int, default=2000)
    p.add_argument('--pool-factor', type=int, default=10, help='Preselect limit*factor; more increases chance of filling filtered pool')
    p.add_argument('--sample-rate', type=float, default=0.1963)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-question-chars', type=int, default=12)
    p.add_argument('--max-question-chars', type=int, default=2000)
    p.add_argument('--min-reference-chars', type=int, default=15)
    p.add_argument('--fuzzy', action='store_true', help='Optional conservative near-copy check. Requires rapidfuzz')
    p.add_argument('--fuzzy-threshold', type=float, default=0.93)
    args = p.parse_args()
    prepare(args)


if __name__ == '__main__': main()
