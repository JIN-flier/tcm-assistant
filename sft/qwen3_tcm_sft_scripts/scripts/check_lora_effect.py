#!/usr/bin/env python3
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = '患者食欲不振，饭后腹胀，大便溏薄，神疲乏力，面色萎黄，舌淡苔白。请从中医角度进行辨证分析。'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--prompt', default=DEFAULT_PROMPT)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
        attn_implementation='sdpa'
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        with model.disable_adapter():
            base_logits = model(**inputs).logits[:, -1, :].float()
        lora_logits = model(**inputs).logits[:, -1, :].float()

    diff = lora_logits - base_logits
    print('mean abs diff:', diff.abs().mean().item())
    print('max abs diff:', diff.abs().max().item())
    print('L2 norm:', diff.norm().item())

    base_id = base_logits.argmax(-1).item(); lora_id = lora_logits.argmax(-1).item()
    print('Base top-1:', base_id, repr(tokenizer.decode([base_id])))
    print('LoRA top-1:', lora_id, repr(tokenizer.decode([lora_id])))

    def topk(logits, title):
        values, ids = torch.topk(logits[0], k=10)
        print('\n' + title)
        for value, idx in zip(values.tolist(), ids.tolist()):
            print(f'{idx:8d} logit={value:9.4f} {repr(tokenizer.decode([idx]))}')

    topk(base_logits, 'Base top-10')
    topk(lora_logits, 'LoRA top-10')


if __name__ == '__main__':
    main()
