#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Infinity
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from infinity.dataset.RS_datasets import get_class2label
from infinity.utils.downstream_metrics import (
    build_head_tail_split,
    build_wandb_payload,
    compute_classification_metrics,
)


def infer_dataset_name(train_path: str) -> str:
    path = train_path.lower()
    if "dior" in path:
        return "dior"
    if "dota" in path:
        return "dota"
    if "fgsc" in path:
        return "fgsc23"
    raise ValueError(f"Cannot infer dataset name from train path: {train_path}")


def parse_pred_csv_pairs(pairs: List[str]) -> List[Tuple[str, str]]:
    # 兼容 argparse 传入 str 的情况（当前 --pred_csv 是 type=str）
    if isinstance(pairs, str):
        pairs = [pairs]

    out = []
    for idx, p in enumerate(pairs):
        p = p.strip()
        if not p:
            continue

        # 支持 name=path
        if "=" in p:
            name, path = p.split("=", 1)
            name = name.strip()
            path = path.strip()
            if not name or not path:
                raise ValueError(f"Invalid --pred_csv item '{p}'")
            out.append((name, path))
        else:
            # 支持仅传路径：自动用文件名作为实验名
            out.append((Path(p).stem or f"exp_{idx+1}", p))

    if not out:
        raise ValueError("No valid --pred_csv item found.")
    return out


def load_train_counts(train_count_csv: str) -> Dict[int, int]:
    """
    CSV schema:
      class_id,train_count
      0,123
      1,456
    """
    df = pd.read_csv(train_count_csv)
    if "class_id" not in df.columns or "train_count" not in df.columns:
        raise ValueError(f"{train_count_csv} must contain columns: class_id, train_count")
    return {int(r["class_id"]): int(r["train_count"]) for _, r in df.iterrows()}


def maybe_init_wandb(args) -> object:
    if not args.wandb_project:
        return None
    import wandb

    mode = "online" if int(args.wandb_online) == 1 else "offline"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        name=args.wandb_run_name if args.wandb_run_name else None,
        mode=mode,
        config={
            "task": "downstream_cls_summary",
            "dataset_train_path": args.train_path,
            "tail_ratio": args.tail_ratio,
            "y_true_col": args.y_true_col,
            "y_pred_col": args.y_pred_col,
        },
    )
    return wandb

# python tools/summarize_downstream_cls.py --dataset_name fgsc
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize downstream classification metrics for multiple experiments."
    )
    parser.add_argument("--dataset_name", type=str, default='FGSC', help="Dataset name (e.g., 'dior', 'dota', 'fgsc'). If not provided, it will be inferred from the train_path.")
    parser.add_argument(
        "--pred_csv",
        type=str,
        default="/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/LT-Uncertainty/dataset/test_results.csv",
        help="Format: exp_name=/path/to/pred.csv (repeat for 4 experiments).",
    )
    parser.add_argument("--y_true_col", type=str, default="label")
    parser.add_argument("--y_pred_col", type=str, default="pred")
    parser.add_argument("--tail_ratio", type=float, default=0.5)
    parser.add_argument(
        "--tail_classes",
        type=str,
        default="",
        help="Optional comma-separated class ids. If provided, overrides tail_ratio split.",
    )
    parser.add_argument("--out_dir", type=str, default='./outputs/downstream_cls_summary')

    # optional wandb logging
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_online", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()
    args.train_path = f"/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/{args.dataset_name}/train"
    args.train_count_csv = f"/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/outputs/Train_Count/{args.dataset_name}_train_counts.csv"

    # DIY
    args.pred_csv = "/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/LT-Uncertainty/dataset/results/test_results_99.csv"


    os.makedirs(args.out_dir, exist_ok=True)
    pred_items = parse_pred_csv_pairs(args.pred_csv)

    dataset_name = infer_dataset_name(args.train_path)
    class2label = get_class2label(dataset_name)  # {class_name: class_id}
    class_ids = sorted([int(v) for v in class2label.values()])
    label_to_name = {int(v): k for k, v in class2label.items()}

    train_counts = load_train_counts(args.train_count_csv)
    for cid in class_ids:
        train_counts.setdefault(cid, 0)

    tail_classes = None
    if args.tail_classes.strip():
        tail_classes = [int(x) for x in args.tail_classes.split(",") if x.strip()]
    split = build_head_tail_split(train_counts, tail_ratio=args.tail_ratio, tail_classes=tail_classes)

    wandb = maybe_init_wandb(args)

    summary_rows: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []

    for exp_name, csv_path in pred_items:
        df = pd.read_csv(csv_path)
        if args.y_true_col not in df.columns or args.y_pred_col not in df.columns:
            raise ValueError(
                f"{csv_path} must contain columns '{args.y_true_col}' and '{args.y_pred_col}'"
            )
        y_true = df[args.y_true_col].astype(int).tolist()
        y_pred = df[args.y_pred_col].astype(int).tolist()

        metrics, per_cls = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            class_ids=class_ids,
            train_counts=train_counts,
            split=split,
        )

        row = {"experiment": exp_name, **metrics}
        summary_rows.append(row)

        for d in per_cls:
            per_class_rows.append(
                {
                    "experiment": exp_name,
                    "class_id": int(d["class_id"]),
                    "class_name": label_to_name.get(int(d["class_id"]), str(d["class_id"])),
                    "train_count": int(d["train_count"]),
                    "test_count": int(d["test_count"]),
                    "correct": int(d["correct"]),
                    "class_acc": float(d["class_acc"]),
                    "group": "tail" if int(d["class_id"]) in split.tail_classes else "head",
                }
            )

        if wandb is not None:
            payload = build_wandb_payload(metrics, prefix=f"Summary/{exp_name}")
            wandb.log(payload)

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    split_info = {
        "head_classes": split.head_classes,
        "tail_classes": split.tail_classes,
    }

    summary_csv = os.path.join(args.out_dir, "summary_metrics.csv")
    per_class_csv = os.path.join(args.out_dir, "per_class_metrics.csv")
    split_json = os.path.join(args.out_dir, "head_tail_split.json")

    summary_df.to_csv(summary_csv, index=False)
    per_class_df.to_csv(per_class_csv, index=False)
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    print(f"[DONE] summary csv: {summary_csv}")
    print(f"[DONE] per-class csv: {per_class_csv}")
    print(f"[DONE] split json: {split_json}")
    print(summary_df.to_string(index=False))

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
