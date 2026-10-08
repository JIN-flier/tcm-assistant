#!/usr/bin/env python3
import argparse
import json
import math
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--top', type=int, default=50)
    args = p.parse_args()

    with open(f'{args.adapter}/adapter_config.json', encoding='utf-8') as f:
        cfg = json.load(f)
    r = int(cfg['r']); alpha = float(cfg['lora_alpha'])
    scaling = alpha / math.sqrt(r) if cfg.get('use_rslora') else alpha / r
    print(f'r={r}, alpha={alpha}, scaling={scaling}')

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True
    )
    lora = load_file(f'{args.adapter}/adapter_model.safetensors')
    modules = dict(model.named_modules())
    results = []

    for name, A in lora.items():
        if not name.endswith('.lora_A.weight'):
            continue
        prefix = name.removesuffix('.lora_A.weight')
        bname = prefix + '.lora_B.weight'
        if bname not in lora:
            continue
        module_name = prefix.replace('base_model.model.', '', 1)
        module = modules.get(module_name)
        if module is None or not hasattr(module, 'weight'):
            continue
        W = module.weight.detach().float()
        delta = (lora[bname].float() @ A.float()) * scaling
        w_norm = torch.linalg.vector_norm(W).item()
        delta_norm = torch.linalg.vector_norm(delta).item()
        ratio = delta_norm / w_norm if w_norm > 0 else float('nan')
        results.append((ratio, delta_norm, w_norm, module_name))

    results.sort(reverse=True)
    print(f"{'module':90s}{'dW/W':>12s}{'||dW||':>12s}{'||W||':>12s}")
    for ratio, dnorm, wnorm, name in results[:args.top]:
        print(f'{name:90s}{ratio:12.6f}{dnorm:12.4f}{wnorm:12.4f}')

    if results:
        t = torch.tensor([x[0] for x in results])
        print('\nModules:', len(results))
        print('Mean dW/W:', t.mean().item())
        print('Median dW/W:', t.median().item())
        print('Max dW/W:', t.max().item())


if __name__ == '__main__':
    main()
