#!/usr/bin/env python3
"""Optional gold-labeled 4-choice benchmark: forced-choice next-token accuracy.

Input JSONL: {"id":"1", "question":"...", "A":"...", "B":"...", "C":"...", "D":"...", "answer":"B"}
The benchmark MUST be independent of SFT; this script does not ship test questions.
"""
import argparse
import json
from pathlib import Path


def macro_f1(y_true, y_pred, labels='ABCD'):
    f1 = []
    for c in labels:
        tp = sum(a == c and b == c for a, b in zip(y_true, y_pred))
        fp = sum(a != c and b == c for a, b in zip(y_true, y_pred))
        fn = sum(a == c and b != c for a, b in zip(y_true, y_pred))
        denom = 2*tp + fp + fn
        f1.append(2*tp/denom if denom else 0)
    return sum(f1)/len(f1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--max-input-tokens', type=int, default=2048)
    args = p.parse_args()
    if not args.input.is_file(): p.error(f'Missing {args.input}')
    if args.output.exists(): p.error(f'Output exists: {args.output}')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = {}
    for c in 'ABCD':
        enc = tokenizer.encode(c, add_special_tokens=False)
        if len(enc) != 1:
            raise SystemExit(f'Cannot use one-token forced choice; tokenization for {c}: {enc}')
        ids[c] = enc[0]
    if len(set(ids.values())) != 4: raise SystemExit('Option IDs not distinct')
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype='auto', device_map='auto',
        trust_remote_code=True, attn_implementation='sdpa')
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    device = model.get_input_embeddings().weight.device
    from contextlib import nullcontext

    def choose(inputs, base_model):
        context = model.disable_adapter() if base_model else nullcontext()
        with torch.inference_mode(), context:
            logits = model(**inputs, use_cache=False).logits[0, -1, :]
            return max('ABCD', key=lambda k: logits[ids[k]].item())

    y, b, s = [], [], []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding='utf-8') as inp, args.output.open('w', encoding='utf-8') as out:
        for line in inp:
            if not line.strip(): continue
            r = json.loads(line)
            label = str(r.get('answer', '')).strip().upper()
            if label not in 'ABCD' or len(label) != 1 or any(c not in r for c in 'ABCD'):
                raise ValueError(f"Invalid answer/options at id={r.get('id')}")
            question = ("下面是一道医学单项选择题。只需回答正确选项的一个大写英文字母 A、B、C 或 D，勿作解释。\n"
                        f"题目：{r['question']}\n" + '\n'.join(f"{c}. {r[c]}" for c in 'ABCD') + '\n正确选项：')
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': question}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
            if inputs['input_ids'].shape[-1] > args.max_input_tokens:
                print('skipped long question:', r.get('id')); continue
            inputs = {k: v.to(device) for k, v in inputs.items()}
            base_pred, sft_pred = choose(inputs, True), choose(inputs, False)
            y.append(label); b.append(base_pred); s.append(sft_pred)
            out.write(json.dumps({'id': r.get('id'), 'answer': label, 'base': base_pred,
                                  'sft': sft_pred}, ensure_ascii=False) + '\n')
            out.flush()
            if len(y) % 50 == 0: print('MCQ evaluated:', len(y), flush=True)
    if not y: raise SystemExit('No valid MCQ evaluated')
    # McNemar discordants: Base only correct vs SFT only correct.
    base_only = sum(bp == gold and sp != gold for gold, bp, sp in zip(y, b, s))
    sft_only = sum(bp != gold and sp == gold for gold, bp, sp in zip(y, b, s))
    summary = dict(n=len(y), base_accuracy=sum(a == c for a, c in zip(y, b))/len(y),
                   sft_accuracy=sum(a == c for a, c in zip(y, s))/len(y),
                   base_macro_f1=macro_f1(y, b), sft_macro_f1=macro_f1(y, s),
                   base_only_correct=base_only, sft_only_correct=sft_only,
                   protocol='forced single-token A/B/C/D next-token logits, not free-form generation')
    report = args.output.with_suffix('.summary.json')
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Saved:', report)


if __name__ == '__main__': main()
