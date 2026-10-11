#!/usr/bin/env python3
"""Paired deterministic inference on held-out MedChatZH with one Base+LoRA model."""
import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--max-input-tokens', type=int, default=2048)
    p.add_argument('--dtype', choices=['auto', 'bf16', 'fp16'], default='auto')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.max_new_tokens <= 0 or args.max_input_tokens <= 0:
        p.error('token lengths must be positive')
    if not args.input.exists(): p.error(f'Missing eval input: {args.input}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_file = args.output.with_suffix('.meta.json')
    meta = dict(model=str(Path(args.model).resolve()), adapter=str(Path(args.adapter).resolve()),
                input_sha256=sha256_file(args.input), max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens, dtype=args.dtype, do_sample=False,
                enable_thinking=False, attn_implementation='sdpa')
    if args.output.exists() and not args.resume:
        p.error('Output exists. Pass --resume to continue, or choose a new --output.')
    if args.resume and args.output.exists():
        if not meta_file.exists() or json.loads(meta_file.read_text()) != meta:
            p.error('Cannot resume: generation settings/model/checkpoint/input differ from recorded metadata')
    elif args.resume and meta_file.exists() and not args.output.exists():
        p.error('Stale metadata without output; remove the metadata file before re-running')
    else:
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    done = set()
    if args.resume and args.output.exists():
        with args.output.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): done.add(str(json.loads(line)['id']))
    # Only import heavy GPU dependencies after CLI and output validation.
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if args.dtype == 'bf16': dt = torch.bfloat16
    elif args.dtype == 'fp16': dt = torch.float16
    else: dt = 'auto'
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dt, device_map='auto', trust_remote_code=True,
        attn_implementation='sdpa')
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    device = model.get_input_embeddings().weight.device
    is_cuda = device.type == 'cuda'

    def run(question, disabled):
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': question}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inp = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        n_input = inp['input_ids'].shape[-1]
        if n_input > args.max_input_tokens:
            return None, n_input
        inp = {k: v.to(device) for k, v in inp.items()}
        ctx = model.disable_adapter() if disabled else nullcontext()
        if is_cuda: torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), ctx:
            result = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, use_cache=True,
                                    pad_token_id=tokenizer.eos_token_id)
        if is_cuda: torch.cuda.synchronize()
        duration = time.perf_counter() - start
        response_ids = result[0, n_input:].tolist()
        return {'answer': tokenizer.decode(response_ids, skip_special_tokens=True).strip(),
                'new_tokens': len(response_ids), 'seconds': duration,
                'tokens_per_second': len(response_ids) / max(duration, 1e-8),
                'hit_max_tokens': len(response_ids) >= args.max_new_tokens}, n_input

    read = skipped = 0
    with args.input.open('r', encoding='utf-8') as f, args.output.open('a', encoding='utf-8') as out:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            if str(row['id']) in done: continue
            question = row['question']
            # Alternate order to avoid systematic warmup-related latency asymmetry.
            first_is_base = read % 2 == 0
            one, n_in = run(question, disabled=first_is_base)
            if one is None:
                skipped += 1
                print(f"skip length={n_in} id={row['id']}", flush=True)
                continue
            two, _ = run(question, disabled=not first_is_base)
            base_res, lora_res = (one, two) if first_is_base else (two, one)
            output = {**row, 'input_tokens': n_in, 'base': base_res, 'sft': lora_res}
            out.write(json.dumps(output, ensure_ascii=False) + '\n')
            out.flush()
            read += 1
            if read % 20 == 0: print(f'generated={read}, skipped_long={skipped}', flush=True)
    print(f'Finished: generated {read} new pairs, skipped {skipped} overlong inputs; {args.output}')


if __name__ == '__main__': main()
