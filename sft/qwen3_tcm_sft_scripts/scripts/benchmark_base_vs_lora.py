#!/usr/bin/env python3
import argparse
import json
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

QUESTIONS = [
    '患者女性，42岁，近一个月容易疲倦，食欲下降，饭后腹胀，大便偏溏，舌淡苔白。请从中医辨证角度进行分析，并说明判断依据。不需要给出具体药物剂量。',
    '请介绍四君子汤的传统组成、主要功效和经典适应证。这里只进行中医方剂知识说明，不针对具体患者开方。',
    '患者只说“最近经常头晕”，目前没有提供更多资料。作为问诊助手，你下一步应该重点询问哪些信息？',
    '请用简单的语言解释哈希表的工作原理，并说明发生哈希冲突时常见的两种处理方法。',
]


def build_inputs(tokenizer, model, question):
    messages = [{'role': 'user', 'content': question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors='pt')
    return {k: v.to(model.device) for k, v in inputs.items()}


def timed_generate(model, tokenizer, question, max_new_tokens, disable_adapter=False):
    inputs = build_inputs(tokenizer, model, question)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    ctx = model.disable_adapter() if disable_adapter else nullcontext()
    pieces = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    def worker():
        with torch.inference_mode(), ctx:
            model.generate(**kwargs)

    thread = threading.Thread(target=worker)
    thread.start()
    first_time = None
    for piece in streamer:
        if piece and first_time is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            first_time = time.perf_counter()
        pieces.append(piece)
    thread.join()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    answer = ''.join(pieces)
    tokens = len(tokenizer.encode(answer, add_special_tokens=False))
    ttft = (first_time - start) if first_time is not None else (end - start)
    decode_time = max(end - (first_time or start), 1e-9)
    return {
        'question': question,
        'answer': answer,
        'output_tokens': tokens,
        'ttft_seconds': ttft,
        'total_seconds': end - start,
        'decode_tokens_per_second': tokens / decode_time,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--output', default='benchmark_results.json')
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation='sdpa',
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    # Warm up once with adapter enabled.
    _ = timed_generate(model, tokenizer, '请回答：1+1等于多少？', 8, disable_adapter=False)

    output = {'base': [], 'lora': []}
    for q in QUESTIONS:
        print('\n' + '=' * 80)
        print('QUESTION:', q)
        base_result = timed_generate(model, tokenizer, q, args.max_new_tokens, disable_adapter=True)
        lora_result = timed_generate(model, tokenizer, q, args.max_new_tokens, disable_adapter=False)
        output['base'].append(base_result)
        output['lora'].append(lora_result)

        print('\n[BASE]\n', base_result['answer'])
        print(f"BASE: TTFT={base_result['ttft_seconds']:.3f}s, total={base_result['total_seconds']:.3f}s, decode={base_result['decode_tokens_per_second']:.2f} tok/s")
        print('\n[LoRA]\n', lora_result['answer'])
        print(f"LoRA: TTFT={lora_result['ttft_seconds']:.3f}s, total={lora_result['total_seconds']:.3f}s, decode={lora_result['decode_tokens_per_second']:.2f} tok/s")

    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
