#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', help='Any checkpoint directory from the run')
    args = p.parse_args()
    state_path = Path(args.checkpoint) / 'trainer_state.json'
    with state_path.open(encoding='utf-8') as f:
        state = json.load(f)
    print('global_step:', state.get('global_step'))
    print('epoch:', state.get('epoch'))
    print('best_metric:', state.get('best_metric'))
    print('best_model_checkpoint:', state.get('best_model_checkpoint'))

    eval_logs = [x for x in state.get('log_history', []) if 'eval_loss' in x]
    if eval_logs:
        print('\nLast eval records:')
        for x in eval_logs[-10:]:
            print('step=', x.get('step'), 'eval_loss=', x.get('eval_loss'), 'epoch=', x.get('epoch'))


if __name__ == '__main__':
    main()
