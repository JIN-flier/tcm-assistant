#!/usr/bin/env python3
import argparse
from safetensors.torch import load_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--top', type=int, default=20)
    args = p.parse_args()
    path = f'{args.checkpoint}/adapter_model.safetensors'
    state = load_file(path)
    total_params = 0; zero_tensors = 0; stats = []
    for name, tensor in state.items():
        t = tensor.float(); total_params += t.numel()
        absmax = t.abs().max().item(); meanabs = t.abs().mean().item(); norm = t.norm().item()
        if absmax == 0:
            zero_tensors += 1
        stats.append((absmax, meanabs, norm, name, tuple(t.shape)))
    print('tensor count:', len(state))
    print('LoRA parameter count:', f'{total_params:,}')
    print('all-zero tensors:', zero_tensors)
    print(f'\nTop {args.top} tensors by max abs value:')
    for absmax, meanabs, norm, name, shape in sorted(stats, reverse=True)[:args.top]:
        print(f'{name:90s} shape={str(shape):20s} max={absmax:.6e} mean={meanabs:.6e} norm={norm:.6e}')


if __name__ == '__main__':
    main()
