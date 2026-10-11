#!/usr/bin/env python3
"""Report proxy text metrics and, if available, blinded judge/human clinical ratings."""
import argparse
import csv
import hashlib
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path


def get_rows(path):
    with Path(path).open(encoding='utf-8') as fh:
        for line in fh:
            if line.strip(): yield json.loads(line)


def chars(text):
    return re.sub(r'\s+', '', text).strip()


def char_bigram_f1(reference, prediction):
    a, b = chars(reference), chars(prediction)
    if len(a) < 2 or len(b) < 2: return 0.0
    a = Counter(a[i:i+2] for i in range(len(a)-1))
    b = Counter(b[i:i+2] for i in range(len(b)-1))
    overlap = sum((a & b).values())
    return 2 * overlap / (sum(a.values()) + sum(b.values()))


def char_rouge_l(reference, prediction):
    from rapidfuzz.distance import LCSseq
    a, b = chars(reference), chars(prediction)
    if not a or not b: return 0.0
    common = LCSseq.similarity(a, b)
    return 2 * common / (len(a) + len(b))


def mean(xs):
    return statistics.fmean(xs) if xs else None


def ci_diff(bases, sfts, bootstrap=1000, seed=42):
    """Paired, percentile bootstrap CI of mean(sft-base)."""
    diffs = [s-b for b, s in zip(bases, sfts)]
    if not diffs: return None
    n = len(diffs); rng = random.Random(seed)
    stats = []
    for _ in range(bootstrap):
        stats.append(sum(diffs[rng.randrange(n)] for _ in range(n))/n)
    stats.sort()
    return [stats[int(.025 * (bootstrap-1))], stats[int(.975 * (bootstrap-1))]]


def sanitize_csv(text):
    text = str(text)
    return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) else text


def blind_order(id_, seed):
    return int.from_bytes(hashlib.blake2b(f'{seed}:{id_}'.encode(), digest_size=8).digest(), 'big') % 2 == 1


def make_review(rows, csv_out, limit, seed):
    shuffled = sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).digest())[:limit]
    csv_out = Path(csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    key_out = csv_out.with_suffix('.key.jsonl')
    header = ['id', 'question', 'reference', 'answer_A', 'answer_B', 'winner',
              'A_correctness', 'B_correctness', 'A_coverage', 'B_coverage', 'A_unsafe', 'B_unsafe', 'notes']
    if csv_out.exists() or key_out.exists():
        raise SystemExit('Blind review files exist; choose a different export path, or remove them explicitly')
    with csv_out.open('w', newline='', encoding='utf-8-sig') as out, key_out.open('w', encoding='utf-8') as key:
        w = csv.DictWriter(out, fieldnames=header)
        w.writeheader()
        for r in shuffled:
            swap = blind_order(str(r['id']), seed)
            answers = (r['sft']['answer'], r['base']['answer']) if swap else (r['base']['answer'], r['sft']['answer'])
            w.writerow(dict(id=r['id'], question=sanitize_csv(r['question']),
                            reference=sanitize_csv(r['reference']),
                            answer_A=sanitize_csv(answers[0]), answer_B=sanitize_csv(answers[1])))
            key.write(json.dumps({'id': r['id'], 'A': 'sft' if swap else 'base',
                                  'B': 'base' if swap else 'sft'}, ensure_ascii=False)+'\n')
    print('Blind review CSV:', csv_out)
    print('SECRET mapping (do not share with reviewers):', key_out)


def parse_bool(s):
    s = str(s).strip().lower()
    if s in ('true', '1', 'yes', 'y', '是'): return True
    if s in ('false', '0', 'no', 'n', '否'): return False
    raise ValueError(f'Invalid Boolean for safety flag: {s}')


def load_human(csv_path, key_path):
    keys = {str(r['id']): r for r in get_rows(key_path)}
    result = {}
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            key = keys.get(row['id']); choice = row.get('winner', '').strip().upper()
            if not key or choice not in ('A', 'B', 'TIE'): continue
            vals = {}
            for col in ('A', 'B'):
                rec = {}
                for k in ('correctness', 'coverage'):
                    raw = row.get(f'{col}_{k}', '').strip()
                    if raw:
                        score = int(raw)
                        if not 1 <= score <= 5: raise ValueError(f'Score outside 1..5, {row["id"]}')
                        rec[k] = score
                raw = row.get(f'{col}_unsafe', '').strip()
                if raw: rec['unsafe'] = parse_bool(raw)
                vals[key[col]] = rec
            result[row['id']] = dict(id=row['id'], winner=key[choice] if choice in 'AB' else 'tie', **vals)
    return result


def evaluate_ratings(rows, ratings):
    valid = [ratings[str(row['id'])] for row in rows if str(row['id']) in ratings]
    if not valid: return None
    base_wins = sum(v['winner'] == 'base' for v in valid)
    sft_wins = sum(v['winner'] == 'sft' for v in valid)
    ties = sum(v['winner'] == 'tie' for v in valid)
    scores = {'sample_count': len(valid), 'base_wins': base_wins, 'sft_wins': sft_wins,
              'ties': ties, 'sft_win_rate_with_half_ties': (sft_wins + .5 * ties) / len(valid)}
    for k in ('correctness', 'coverage', 'unsafe'):
        pairs = [(v['base'][k], v['sft'][k]) for v in valid
                 if k in v.get('base', {}) and k in v.get('sft', {})]
        if pairs:
            b, s = zip(*pairs)
            scores[k] = {'n': len(pairs), 'base': mean(b), 'sft': mean(s),
                         'difference_sft_minus_base': mean([y-x for x,y in pairs]),
                         'ci95': ci_diff(b, s)}
    return scores


def report(args):
    rows = list(get_rows(args.input))
    if not rows: raise SystemExit('No model generations found')
    if args.export_review:
        make_review(rows, args.export_review, args.review_limit, args.seed)
    result = {'n': len(rows), 'metric_warning': (
        'ROUGE-L and character bigram F1 are text-overlap proxies, NOT medical accuracy. '
        'MedChatZH references are not verified clinical ground truth. '
        'Judge/human assessment measures reviewer judgments, not validated clinical correctness.')}
    metrics = {
        'char_rouge_l_f1': char_rouge_l,
        'char_bigram_f1': char_bigram_f1,
    }
    for label, fn in metrics.items():
        base = [fn(row['reference'], row['base']['answer']) for row in rows]
        sft = [fn(row['reference'], row['sft']['answer']) for row in rows]
        result[label] = {'base': mean(base), 'sft': mean(sft),
                         'delta': mean([s - b for b, s in zip(base, sft)]),
                         'ci95': ci_diff(base, sft)}
    for field in ('new_tokens', 'seconds', 'tokens_per_second'):
        result[field] = {'base': mean([row['base'][field] for row in rows]),
                         'sft': mean([row['sft'][field] for row in rows])}
    result['exact_same_answer_fraction'] = sum(r['base']['answer'] == r['sft']['answer'] for r in rows) / len(rows)
    result['base_truncated_fraction'] = mean([bool(r['base']['hit_max_tokens']) for r in rows])
    result['sft_truncated_fraction'] = mean([bool(r['sft']['hit_max_tokens']) for r in rows])
    if args.judge:
        result['llm_judge'] = evaluate_ratings(rows, {str(r['id']): r for r in get_rows(args.judge)})
    if args.human_review:
        if not args.review_key: raise SystemExit('--review-key required with --human-review')
        result['human_review'] = evaluate_ratings(rows, load_human(args.human_review, args.review_key))
    prefix = Path(args.output_prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    def nfmt(x): return 'N/A' if x is None else f'{x:.4f}'
    def dc(d): return nfmt(d['delta']) + ' [' + ', '.join(nfmt(v) for v in d['ci95']) + ']'
    lines = ['# Qwen3-8B Base vs LoRA：MedChatZH 剩余数据评测', '',
             f"- 成功生成的成对样本：**{len(rows)}**", f"- 完全相同回答比例：**{result['exact_same_answer_fraction']:.2%}**", '',
             '## 参考答案文本重叠度（不是医学准确率）', '',
             '| 指标 | Base | SFT | Δ SFT−Base [95% 配对 bootstrap CI] |', '|---|---:|---:|---:|']
    for label in metrics:
        d = result[label]
        lines.append(f"| {label} | {nfmt(d['base'])} | {nfmt(d['sft'])} | {dc(d)} |")
    lines += ['', '## 生成开销', '', '| 指标 | Base | SFT |', '|---|---:|---:|']
    for field in ('new_tokens', 'seconds', 'tokens_per_second'):
        d = result[field]
        lines.append(f"| {field} | {nfmt(d['base'])} | {nfmt(d['sft'])} |")
    for section, label in [('llm_judge', '独立大模型匿名评审'), ('human_review', '人工匿名评审')]:
        data = result.get(section)
        if not data: continue
        lines += ['', f'## {label}', '', f"- 已评价样本：{data['sample_count']}",
                  f"- SFT胜 / Base胜 / 平局：{data['sft_wins']} / {data['base_wins']} / {data['ties']}",
                  f"- SFT胜率（平局计0.5）：**{data['sft_win_rate_with_half_ties']:.2%}**", '',
                  '| 指标 | Base | SFT | Δ [95% 配对 bootstrap CI] |', '|---|---:|---:|---:|']
        for k in ('correctness', 'coverage', 'unsafe'):
            if k not in data: continue
            d = data[k]
            lines.append(f"| {k} (n={d['n']}) | {nfmt(d['base'])} | {nfmt(d['sft'])} | "
                         f"{nfmt(d['difference_sft_minus_base'])} [{', '.join(nfmt(v) for v in d['ci95'])}] |")
    lines += ['', '## 解释限制', '',
              '- ROUGE-L 与字符二元组 F1 仅表示参考文本相似度，不能称作诊断准确率。',
              '- MedChatZH 包含多种医疗/通用内容，此汇总不能直接解释为中医辨证准确率。',
              '- 来源数据的参考回答不一定医学正确，医学质量需要专业人员盲评。',
              '- 自动评审可能存在评判偏差、安全性漏判；请针对危险建议进行人工复核。',
              '- 去重程序不会可靠捕获语义改写，需对候选问答另做审查。',
              '- 置信区间体现这组样本的抽样不确定性，不消除系统性偏差。', '']
    prefix.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved:', prefix.with_suffix('.json'), prefix.with_suffix('.md'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output-prefix', required=True)
    p.add_argument('--judge', type=Path, help='judge_medchat_pairs.py output JSONL')
    p.add_argument('--human-review', type=Path, help='Completed blind review CSV')
    p.add_argument('--review-key', type=Path, help='Blind-review .key.jsonl mapping (not shared with raters)')
    p.add_argument('--export-review', type=Path, help='Export unfilled randomized, blinded human-review CSV')
    p.add_argument('--review-limit', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    report(p.parse_args())


if __name__ == '__main__': main()
