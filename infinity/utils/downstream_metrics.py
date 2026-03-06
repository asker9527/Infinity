from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
 

@dataclass
class GroupSplit:
    head_classes: List[int]
    tail_classes: List[int]


def build_head_tail_split(
    train_counts: Dict[int, int],
    tail_ratio: float = 0.5,
    tail_classes: Optional[Iterable[int]] = None,
) -> GroupSplit:
    """
    Build head/tail split by train class frequency.

    - If tail_classes is provided, use it directly.
    - Otherwise, sort classes by train count ascending and assign the bottom tail_ratio classes to tail.
    """
    all_classes = sorted(train_counts.keys())
    if tail_classes is not None:
        tail = sorted(set(int(x) for x in tail_classes))
        head = sorted([c for c in all_classes if c not in tail])
        return GroupSplit(head_classes=head, tail_classes=tail)

    sorted_by_count = sorted(all_classes, key=lambda c: train_counts.get(c, 0))
    n_cls = len(sorted_by_count)
    n_tail = max(1, int(round(n_cls * float(tail_ratio))))
    tail = sorted(sorted_by_count[:n_tail])
    head = sorted([c for c in sorted_by_count if c not in tail])
    return GroupSplit(head_classes=head, tail_classes=tail)


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0
    return float(valid.mean())


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_ids: Sequence[int],
    train_counts: Dict[int, int],
    split: GroupSplit,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Return:
      summary metrics dict
      per-class metrics list
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    class_ids = [int(c) for c in class_ids]
    per_class: List[Dict[str, float]] = []
    total_correct = int((y_true == y_pred).sum())
    total_num = int(y_true.size)

    for c in class_ids:
        idx = y_true == c
        n = int(idx.sum())
        correct = int((y_pred[idx] == c).sum()) if n > 0 else 0
        acc = (correct / n * 100.0) if n > 0 else float("nan")
        per_class.append(
            {
                "class_id": c,
                "train_count": int(train_counts.get(c, 0)),
                "test_count": n,
                "correct": correct,
                "class_acc": acc,
            }
        )

    class_acc_map = {int(d["class_id"]): float(d["class_acc"]) for d in per_class}

    def _group_weighted_acc(group: Sequence[int]) -> float:
        if not group:
            return 0.0
        mask = np.isin(y_true, np.array(group, dtype=np.int64))
        n = int(mask.sum())
        if n == 0:
            return 0.0
        correct = int((y_true[mask] == y_pred[mask]).sum())
        return correct / n * 100.0

    def _group_avg_acc(group: Sequence[int]) -> float:
        return _safe_mean([class_acc_map.get(int(c), float("nan")) for c in group])

    metrics = {
        "all_acc": (total_correct / max(1, total_num) * 100.0),
        "all_avg_acc": _safe_mean([d["class_acc"] for d in per_class]),
        "head_acc": _group_weighted_acc(split.head_classes),
        "head_avg_acc": _group_avg_acc(split.head_classes),
        "tail_acc": _group_weighted_acc(split.tail_classes),
        "tail_avg_acc": _group_avg_acc(split.tail_classes),
        "num_samples": float(total_num),
        "num_head_classes": float(len(split.head_classes)),
        "num_tail_classes": float(len(split.tail_classes)),
    }
    return metrics, per_class


def build_wandb_payload(
    metrics: Dict[str, float],
    prefix: str = "Downstream",
) -> Dict[str, float]:
    return {f"{prefix}/{k}": float(v) for k, v in metrics.items()}


def log_to_wandb_if_available(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = "Downstream",
) -> None:
    """
    Optional helper for training loops.
    Example:
        from infinity.utils.downstream_metrics import log_to_wandb_if_available
        log_to_wandb_if_available(metrics, step=epoch, prefix='Eval')
    """
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.log(build_wandb_payload(metrics, prefix=prefix), step=step)
    except Exception:
        # Keep training/evaluation robust even if wandb env is unavailable.
        return
