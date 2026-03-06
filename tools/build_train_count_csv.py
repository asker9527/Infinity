#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
# 确保项目根目录在 sys.path 中以加载 infinity 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Dict

import pandas as pd
from torchvision import datasets

from infinity.dataset.RS_datasets import get_class2label



def infer_dataset_name(train_path: str) -> str:
    path = train_path.lower()
    if "dior" in path:
        return "dior"
    if "dota" in path:
        return "dota"
    if "fgsc" in path:
        return "fgsc23"
    raise ValueError(f"Cannot infer dataset name from train path: {train_path}")

# python tools/build_train_count_csv.py --dataset_name FGSC

def main() -> None:
    parser = argparse.ArgumentParser(description="Build train class count CSV from ImageFolder directory.")
    parser.add_argument("--dataset_name", type=str, default='FGSC', help="Dataset name (e.g., 'dior', 'dota', 'fgsc'). If not provided, it will be inferred from the train_path.")
    args = parser.parse_args()
    args.train_path = f"/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/{args.dataset_name}/train"
    args.out_csv = f"./outputs/Train_Count/{args.dataset_name}_train_counts.csv"

    dataset_name = infer_dataset_name(args.train_path)
    class2label = get_class2label(dataset_name)  # class_name -> class_id: {"non-ship": 0,...}

    ds = datasets.ImageFolder(root=args.train_path)
    counts_by_name: Dict[str, int] = {name: 0 for name in ds.classes}
    for _, idx in ds.samples:
        cls_name = ds.classes[idx]
        counts_by_name[cls_name] = counts_by_name.get(cls_name, 0) + 1

    rows = []
    for class_name, class_id in sorted(class2label.items(), key=lambda kv: kv[1]):
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "train_count": int(counts_by_name.get(str(class_id), 0)),
            }
        )

    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[DONE] {args.out_csv}")


if __name__ == "__main__":
    main()
