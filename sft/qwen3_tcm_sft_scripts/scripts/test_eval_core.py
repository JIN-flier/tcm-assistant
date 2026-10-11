"""No GPU / private data required. Run: python -m unittest discover -s tests -v"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_medchat_eval import stable_probability, normalized, prepare
from report_medchat_eval import char_bigram_f1, char_rouge_l, blind_order, load_human
from judge_medchat_pairs import validate, extract_json


class TestEvalCore(unittest.TestCase):
    def test_medchat_hash_matches_spec(self):
        import hashlib
        text = '问诊问题\n参考回答'
        expected = int.from_bytes(hashlib.blake2b(('42:' + text).encode(), digest_size=8).digest(), 'big') / (2**64 - 1)
        self.assertEqual(stable_probability(text, 42), expected)

    def test_norm(self):
        self.assertEqual(normalized(' 头 晕：三天。'), normalized('头晕:三天'))

    def test_question_leakage(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            raw = root / 'raw.jsonl'; train = root / 'train.jsonl'
            valid = root / 'val.jsonl'; test = root / 'test.jsonl'
            out = root / 'eval.jsonl'
            rows = []
            for i in range(200):
                q = f'这个编号为{i:04d}的临床病例出现发热、头晕，下一步应怎么询问病史？'
                a = f'病例{i:04d}建议仔细询问相关症状的持续时间，筛查危险信号。'
                rows.append({'instruction': q, 'input': '', 'output': a})
            raw.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
            unsampled = [r for r in rows if stable_probability(r['instruction'] + '\n' + r['output'], 42) >= 0.1963]
            self.assertGreater(len(unsampled), 30)
            danger = unsampled[0]['instruction']
            train.write_text(json.dumps({'messages': [
                {'role': 'user', 'content': danger},
                {'role': 'assistant', 'content': '不同参考答案'}]}, ensure_ascii=False) + '\n', encoding='utf-8')
            valid.touch(); test.touch()
            args = Namespace(raw=raw, splits=[train, valid, test], output=out,
                             limit=10, pool_factor=20, sample_rate=.1963, seed=42,
                             min_question_chars=12, max_question_chars=2000,
                             min_reference_chars=15, fuzzy=False, fuzzy_threshold=.93)
            prepare(args)
            data = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(data), 10)
            self.assertNotIn(danger, {r['question'] for r in data})
            self.assertTrue((root / 'eval.audit.json').is_file())

    def test_metric_identity(self):
        self.assertEqual(char_rouge_l('中医辨证', '中医辨证'), 1.0)
        self.assertEqual(char_bigram_f1('中医辨证', '中医辨证'), 1.0)
        self.assertAlmostEqual(char_bigram_f1('中医辨证', '完全无关'), 0)

    def test_judge(self):
        obj = {'winner': 'tie', 'A': {'correctness':3, 'coverage':4, 'unsafe':False},
               'B': {'correctness':4, 'coverage':4, 'unsafe':False}}
        self.assertEqual(validate(extract_json('```json\n' + json.dumps(obj) + '\n```')), obj)
        self.assertEqual(blind_order('test-case', 42), blind_order('test-case', 42))


if __name__ == '__main__': unittest.main()
