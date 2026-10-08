#!/usr/bin/env python3
"""Python alternative to download_datasets.sh."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', required=True)
    args = p.parse_args()
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id='tyang816/MedChatZH', repo_type='dataset',
        local_dir=raw / 'MedChatZH',
        allow_patterns=['MedChatZH_train.json', 'MedChatZH_valid.json', 'README.md'],
    )
    snapshot_download(
        repo_id='ZJUFanLab/TCMChat-dataset-600k', repo_type='dataset',
        local_dir=raw / 'TCMChat', allow_patterns=['sft/**', 'README.md'],
    )
    snapshot_download(
        repo_id='SylvanL/Traditional-Chinese-Medicine-Dataset-SFT', repo_type='dataset',
        local_dir=raw / 'TraditionalTCM',
    )
    snapshot_download(
        repo_id='TigerResearch/sft_zh', repo_type='dataset',
        local_dir=raw / 'General' / 'TigerBot',
        allow_patterns=['tigerbot-alpaca-zh-0.5m.json', 'README.md'],
    )
    print('All datasets downloaded under:', raw)


if __name__ == '__main__':
    main()
