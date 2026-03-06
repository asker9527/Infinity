#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_exp_dirs(exp_dirs: List[str], exp_root: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    if exp_dirs:
        for item in exp_dirs:
            if "=" not in item:
                raise ValueError(f"Invalid --exp_dirs item '{item}', expected name=/path/to/exp")
            name, p = item.split("=", 1)
            result[name.strip()] = Path(p.strip()).resolve()
        return result

    if not exp_root:
        raise ValueError("Either --exp_dirs or --exp_root must be provided.")
    root = Path(exp_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"exp_root not found: {root}")

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "generation_summary.json").exists():
            result[d.name] = d
    if not result:
        raise RuntimeError(f"No experiment dirs found under {root} (expect generation_summary.json).")
    return result


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_table(df: pd.DataFrame, csv_path: Path, tex_path: Optional[Path] = None) -> None:
    df.to_csv(csv_path, index=False)
    if tex_path is not None:
        # Minimal latex export, user can refine in thesis.
        tex = df.to_latex(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
        tex_path.write_text(tex, encoding="utf-8")


def build_generation_efficiency_table(exp_map: Dict[str, Path]) -> pd.DataFrame:
    rows = []
    for exp_name, exp_dir in exp_map.items():
        summary_path = exp_dir / "generation_summary.json"
        if not summary_path.exists():
            print(f"[WARN] missing {summary_path}, skip {exp_name}")
            continue
        s = read_json(summary_path)
        run_time_s = float(s.get("run_time_s", 0.0))
        total_kept = float(s.get("total_kept", 0.0))
        eta = total_kept / (run_time_s / 3600.0) if run_time_s > 0 else 0.0
        rows.append(
            {
                "experiment": exp_name,
                "filter_mode": s.get("filter_mode", ""),
                "budget_mode": s.get("budget_mode", s.get("augment_strategy", "")),
                "total_attempts": int(s.get("total_attempts", 0)),
                "total_kept": int(s.get("total_kept", 0)),
                "pass_rate": float(s.get("pass_rate", 0.0)),
                "run_time_s": run_time_s,
                "mean_gen_time_s": float(s.get("mean_gen_time_s", 0.0)),
                "mean_filter_time_s": float(s.get("mean_filter_time_s", 0.0)),
                "mean_total_time_s": float(s.get("mean_total_time_s", 0.0)),
                "eta_kept_per_hour": eta,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("experiment").reset_index(drop=True)
    return df


def merge_with_downstream_metrics(
    generation_df: pd.DataFrame,
    downstream_summary_csv: Optional[str],
) -> pd.DataFrame:
    if not downstream_summary_csv:
        return generation_df.copy()
    dpath = Path(downstream_summary_csv).resolve()
    if not dpath.exists():
        print(f"[WARN] downstream summary csv not found: {dpath}")
        return generation_df.copy()
    ddf = pd.read_csv(dpath)
    if "experiment" not in ddf.columns:
        print(f"[WARN] downstream summary csv missing 'experiment' col: {dpath}")
        return generation_df.copy()
    merged = pd.merge(generation_df, ddf, on="experiment", how="outer")
    return merged


def plot_budget_entropy(exp_map: Dict[str, Path], out_path: Path) -> None:
    """
    Plot entropy-driven budget (per-class entropy vs assigned dynamic budget) if planner exists.
    """
    # pick first experiment containing planner with class_entropy + dynamic_budget
    for exp_name, exp_dir in exp_map.items():
        summary_path = exp_dir / "generation_summary.json"
        if not summary_path.exists():
            continue
        s = read_json(summary_path)
        planner = s.get("planner", {})
        class_entropy = planner.get("class_entropy", {})
        dyn_budget = planner.get("dynamic_budget", [])
        if not class_entropy or not dyn_budget:
            continue

        # x-axis ordered by class id
        cls_ids = sorted([int(k) for k in class_entropy.keys()])
        entropy_vals = [float(class_entropy[str(k)]) for k in cls_ids]
        if len(dyn_budget) != len(cls_ids):
            # try align by class index length
            n = min(len(dyn_budget), len(cls_ids))
            cls_ids = cls_ids[:n]
            entropy_vals = entropy_vals[:n]
            dyn_budget = dyn_budget[:n]

        x = np.arange(len(cls_ids))
        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax1.bar(x, dyn_budget, alpha=0.6, label="Dynamic Budget")
        ax1.set_ylabel("Generated Samples Budget")
        ax1.set_xlabel("Class ID")
        ax1.set_xticks(x)
        ax1.set_xticklabels(cls_ids, rotation=90)

        ax2 = ax1.twinx()
        ax2.plot(x, entropy_vals, color="tab:red", linewidth=2, label="Class Entropy")
        ax2.set_ylabel("Class Entropy")

        ax1.set_title(f"Entropy-driven Budget Allocation ({exp_name})")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    print("[WARN] No experiment has planner.class_entropy + planner.dynamic_budget; skip budget plot.")


def plot_scale_distribution(exp_map: Dict[str, Path], out_path: Path) -> None:
    """
    Plot chosen scale distribution among accepted samples across experiments.
    """
    rows = []
    for exp_name, exp_dir in exp_map.items():
        rec_path = exp_dir / "generation_records.csv"
        if not rec_path.exists():
            continue
        df = pd.read_csv(rec_path)
        if "accepted" not in df.columns or "chosen_scale_idx" not in df.columns:
            continue
        d = df[df["accepted"] == 1].copy()
        if d.empty:
            continue
        stat = d.groupby("chosen_scale_idx").size().reset_index(name="count")
        stat["experiment"] = exp_name
        rows.append(stat)
    if not rows:
        print("[WARN] No generation_records with chosen_scale_idx found; skip scale distribution plot.")
        return

    all_df = pd.concat(rows, axis=0, ignore_index=True)
    pivot = all_df.pivot_table(
        index="chosen_scale_idx", columns="experiment", values="count", aggfunc="sum", fill_value=0
    )
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Chosen Scale Distribution (Accepted Samples)")
    ax.set_xlabel("Scale Index")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_filter_metric_box(exp_map: Dict[str, Path], out_path: Path) -> None:
    """
    Box-like summary for confidence/entropy/disagreement by accepted vs rejected.
    Implemented with matplotlib boxplot for minimal dependency.
    """
    parts = []
    for exp_name, exp_dir in exp_map.items():
        rec_path = exp_dir / "generation_records.csv"
        if not rec_path.exists():
            continue
        df = pd.read_csv(rec_path)
        required = {"accepted", "confidence", "entropy", "disagreement"}
        if not required.issubset(df.columns):
            continue
        df["experiment"] = exp_name
        parts.append(df[["experiment", "accepted", "confidence", "entropy", "disagreement"]])
    if not parts:
        print("[WARN] No records for filter metrics; skip filter metric plot.")
        return
    data = pd.concat(parts, ignore_index=True)

    # Use all experiments merged to show mechanism gap.
    acc = data[data["accepted"] == 1]
    rej = data[data["accepted"] == 0]
    metrics = ["confidence", "entropy", "disagreement"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, m in enumerate(metrics):
        ax = axes[i]
        vals_acc = acc[m].dropna().to_numpy()
        vals_rej = rej[m].dropna().to_numpy()
        box_data = []
        labels = []
        if len(vals_acc) > 0:
            box_data.append(vals_acc)
            labels.append("Accepted")
        if len(vals_rej) > 0:
            box_data.append(vals_rej)
            labels.append("Rejected")
        if box_data:
            ax.boxplot(box_data, labels=labels, showfliers=False)
        ax.set_title(m)
    fig.suptitle("Filtering Metrics: Accepted vs Rejected")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_tail_gain(per_class_csv: Optional[str], out_path: Path, baseline: str, target: str) -> None:
    """
    Plot per-class accuracy gain: target - baseline.
    Requires per_class_metrics.csv generated by summarize_downstream_cls.py
    """
    if not per_class_csv:
        print("[WARN] per_class_csv not provided; skip tail gain plot.")
        return
    p = Path(per_class_csv).resolve()
    if not p.exists():
        print(f"[WARN] per_class_csv not found: {p}; skip tail gain plot.")
        return
    df = pd.read_csv(p)
    required = {"experiment", "class_id", "class_acc", "group"}
    if not required.issubset(df.columns):
        print(f"[WARN] per_class_csv missing cols {required}; skip tail gain plot.")
        return

    b = df[df["experiment"] == baseline][["class_id", "class_acc"]].rename(columns={"class_acc": "acc_base"})
    t = df[df["experiment"] == target][["class_id", "class_acc", "group"]].rename(columns={"class_acc": "acc_target"})
    m = pd.merge(t, b, on="class_id", how="inner")
    if m.empty:
        print(f"[WARN] No overlapping classes between baseline={baseline} and target={target}; skip tail gain plot.")
        return
    m["gain"] = m["acc_target"] - m["acc_base"]
    m = m.sort_values("gain", ascending=False).reset_index(drop=True)

    colors = ["tab:red" if g == "tail" else "tab:blue" for g in m["group"]]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(len(m)), m["gain"].to_numpy(), color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"Per-class Accuracy Gain: {target} - {baseline}")
    ax.set_xlabel("Class (sorted by gain)")
    ax.set_ylabel("Delta class_acc")
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chapter-5 thesis tables and plots from experiment outputs.")
    parser.add_argument("--exp_root", type=str, default="./outputs/Generated_Results/DOTA", help="Root folder containing experiment dirs.")
    parser.add_argument(
        "--exp_dirs",
        type=str,
        nargs="*",
        default=['var_full=/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/outputs/Generated_Results/DOTA/var_full', 
                 'var_confidence=/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/outputs/Generated_Results/DOTA/var_confidence',
                 'var_entropy=/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/outputs/Generated_Results/DOTA/var_entropy',
                 'var_joint=/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/outputs/Generated_Results/DOTA/var_joint'],
        help="Optional explicit experiment mapping: exp_name=/path/to/exp_dir",
    )
    parser.add_argument(
        "--downstream_summary_csv",
        type=str,
        default="/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/outputs/downstream_cls_summary/summary_metrics.csv",
        help="summary_metrics.csv from tools/summarize_downstream_cls.py",
    )
    parser.add_argument(
        "--per_class_csv",
        type=str,
        default="/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/outputs/downstream_cls_summary/per_class_metrics.csv",
        help="per_class_metrics.csv from tools/summarize_downstream_cls.py",
    )
    parser.add_argument("--baseline_exp", type=str, default="var_full")
    parser.add_argument("--target_exp", type=str, default="var_joint")
    parser.add_argument("--out_dir", type=str, default="./outputs/chap5_tables_and_plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    ensure_dir(fig_dir)
    ensure_dir(tab_dir)

    exp_map = parse_exp_dirs(args.exp_dirs, args.exp_root)
    print(f"[INFO] Found experiments: {list(exp_map.keys())}")

    # Tables
    gen_df = build_generation_efficiency_table(exp_map)
    merged_df = merge_with_downstream_metrics(gen_df, args.downstream_summary_csv)

    save_table(
        gen_df,
        tab_dir / "table_generation_efficiency.csv",
        tab_dir / "table_generation_efficiency.tex",
    )
    save_table(
        merged_df,
        tab_dir / "table_chap5_main_metrics.csv",
        tab_dir / "table_chap5_main_metrics.tex",
    )

    # Plots
    plot_budget_entropy(exp_map, fig_dir / "fig_budget_entropy.png")
    plot_scale_distribution(exp_map, fig_dir / "fig_scale_distribution.png")
    plot_filter_metric_box(exp_map, fig_dir / "fig_filter_metric_box.png")
    plot_tail_gain(args.per_class_csv, fig_dir / "fig_tail_gain.png", args.baseline_exp, args.target_exp)

    print("[DONE] Outputs:")
    print(f"  - Tables: {tab_dir}")
    print(f"  - Figures: {fig_dir}")


if __name__ == "__main__":
    main()

