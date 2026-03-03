"""
Generate synthetic remote-sensing images with Infinity VAR.

New features:
1) Dynamic class budget by model uncertainty entropy (budget_mode=entropy_dynamic)
2) Multi-scale candidate selection in one VAR generation pass
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets as tv_datasets
from torchvision import models, transforms

from tools.diy_tools import get_models
from tools.run_infinity import dynamic_resolution_h_w, gen_one_img, h_div_w_templates
from infinity.dataset.RS_datasets import get_RS_datasets, get_class2label


def set_number_of_samples_per_class(
    nums_per_class: Sequence[int],
    strategy: str = "balance",
    target_num: int = 500,
    fixed_add: int = 100,
    ratio: float = 1.0,
) -> List[int]:
    if strategy == "balance":
        return [max(0, target_num - int(num)) for num in nums_per_class]
    if strategy == "fixed":
        return [max(0, int(fixed_add)) for _ in nums_per_class]
    if strategy == "ratio":
        return [max(0, int(math.ceil(num * ratio))) for num in nums_per_class]
    raise ValueError(f"Unsupported strategy: {strategy}")


def infer_dataset_name(train_path: str, test_path: str) -> str:
    merged = f"{train_path}|{test_path}".lower()
    if "dior" in merged:
        return "dior"
    if "dota" in merged:
        return "dota"
    if "fgsc" in merged:
        return "fgsc23"
    raise ValueError(f"Cannot infer dataset name from paths: {train_path}, {test_path}")


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def tensor_hwc_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unexpected generated image shape: {arr.shape}")
    return arr


class EnsembleClassifier:
    def __init__(
        self,
        ckpt_paths: Sequence[str],
        num_classes: int,
        device: str = "cuda",
        model_name: str = "resnet50",
    ):
        if not ckpt_paths:
            raise ValueError("Classifier checkpoints are required.")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.models = [self._load_one(path, num_classes, model_name) for path in ckpt_paths]
        self.num_classes = num_classes
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _build_model(self, model_name: str, num_classes: int) -> torch.nn.Module:
        if model_name == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            return model
        if model_name == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            return model
        raise ValueError(f"Unsupported classifier model: {model_name}")

    def _load_one(self, ckpt_path: str, num_classes: int, model_name: str) -> torch.nn.Module:
        model = self._build_model(model_name, num_classes)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        clean = {}
        for k, v in ckpt.items():
            nk = k[7:] if k.startswith("module.") else k
            clean[nk] = v
        model.load_state_dict(clean, strict=False)
        model.eval().to(self.device)
        return model

    @torch.no_grad()
    def predict_probs(self, image_hwc_uint8: np.ndarray) -> torch.Tensor:
        x = self.preprocess(image_hwc_uint8).unsqueeze(0).to(self.device)
        probs = []
        for model in self.models:
            logits = model(x)
            probs.append(F.softmax(logits, dim=1))
        return torch.cat(probs, dim=0)  # [T, K]


@dataclass
class FilterThresholds:
    confidence_min: float = 0.50
    entropy_max: float = 0.80
    disagreement_max: float = 0.20


class GeneratedDataFilter:
    def __init__(
        self,
        mode: str,
        thresholds: FilterThresholds,
        classifier: Optional[EnsembleClassifier] = None,
    ):
        self.mode = mode
        self.thresholds = thresholds
        self.classifier = classifier
        if self.mode != "full" and self.classifier is None:
            raise ValueError(f"Mode '{self.mode}' requires classifier checkpoints.")

    @torch.no_grad()
    def evaluate(self, image_hwc_uint8: np.ndarray, target_idx: int) -> Dict[str, float]:
        if self.mode == "full":
            return {
                "accepted": 1.0,
                "pred_idx": float(target_idx),
                "confidence": 1.0,
                "entropy": 0.0,
                "disagreement": 0.0,
                "reason": "full_pass",
            }

        probs_tk = self.classifier.predict_probs(image_hwc_uint8)  # [T, K]
        p_bar = probs_tk.mean(dim=0)
        pred_idx = int(torch.argmax(p_bar).item())
        confidence = float(torch.max(p_bar).item())
        eps = 1e-8
        entropy = float((-(p_bar * torch.log(p_bar + eps)).sum() / math.log(p_bar.numel())).item())
        kls = []
        for t in range(probs_tk.shape[0]):
            p_t = probs_tk[t]
            kl = torch.sum(p_t * (torch.log(p_t + eps) - torch.log(p_bar + eps)))
            kls.append(kl)
        disagreement = float((torch.stack(kls).mean() / math.log(p_bar.numel())).item())

        if pred_idx != target_idx:
            return {
                "accepted": 0.0,
                "pred_idx": float(pred_idx),
                "confidence": confidence,
                "entropy": entropy,
                "disagreement": disagreement,
                "reason": "class_mismatch",
            }

        if self.mode == "confidence":
            ok = confidence >= self.thresholds.confidence_min
            reason = "pass" if ok else "low_confidence"
        elif self.mode == "entropy":
            ok = entropy <= self.thresholds.entropy_max
            reason = "pass" if ok else "high_entropy"
        elif self.mode == "joint":
            ok = (
                confidence >= self.thresholds.confidence_min
                and entropy <= self.thresholds.entropy_max
                and disagreement <= self.thresholds.disagreement_max
            )
            if ok:
                reason = "pass"
            elif confidence < self.thresholds.confidence_min:
                reason = "low_confidence"
            elif entropy > self.thresholds.entropy_max:
                reason = "high_entropy"
            else:
                reason = "high_disagreement"
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return {
            "accepted": float(ok),
            "pred_idx": float(pred_idx),
            "confidence": confidence,
            "entropy": entropy,
            "disagreement": disagreement,
            "reason": reason,
        }


def _allocate_integer_budget(weights: Sequence[float], total_budget: int) -> List[int]:
    total_budget = int(max(0, total_budget))
    if total_budget == 0:
        return [0 for _ in weights]
    w = np.array(weights, dtype=np.float64)
    if np.sum(w) <= 0:
        w = np.ones_like(w)
    raw = total_budget * w / np.sum(w)
    flo = np.floor(raw).astype(int)
    rem = total_budget - int(flo.sum())
    if rem > 0:
        frac_order = np.argsort(-(raw - flo))
        for i in frac_order[:rem]:
            flo[i] += 1
    return flo.tolist()


def estimate_class_entropy(
    train_path: str,
    class2label: Dict[str, int],
    classifier: EnsembleClassifier,
    max_samples_per_class: int = 0,
) -> Dict[int, float]:
    ds = tv_datasets.ImageFolder(root=train_path)
    counts_used: Dict[int, int] = {v: 0 for v in class2label.values()}
    entropy_vals: Dict[int, List[float]] = {v: [] for v in class2label.values()}
    eps = 1e-8
    for img_path, ds_idx in ds.samples:
        class_name = ds.classes[ds_idx]
        if class_name not in class2label:
            continue
        cid = int(class2label[class_name])
        if max_samples_per_class > 0 and counts_used[cid] >= max_samples_per_class:
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        probs_tk = classifier.predict_probs(img)
        p_bar = probs_tk.mean(dim=0)
        entropy = float((-(p_bar * torch.log(p_bar + eps)).sum() / math.log(p_bar.numel())).item())
        entropy_vals[cid].append(entropy)
        counts_used[cid] += 1
    out = {}
    for cid, vals in entropy_vals.items():
        out[cid] = float(np.mean(vals)) if vals else 0.0
    return out


def dynamic_budget_by_entropy(
    base_add_nums: Sequence[int],
    nums_per_class: Sequence[int],
    class_indices: Sequence[int],
    class_entropy: Dict[int, float],
    entropy_mix_imbalance: float,
    entropy_temperature: float,
    budget_total: int,
) -> List[int]:
    max_n = max(max(nums_per_class), 1)
    scores = []
    for cid, n in zip(class_indices, nums_per_class):
        ent = float(class_entropy.get(int(cid), 0.0))
        imb = (max_n - n) / max_n
        score = max(1e-8, ent + float(entropy_mix_imbalance) * imb)
        if entropy_temperature != 1.0:
            score = score ** (1.0 / max(entropy_temperature, 1e-6))
        scores.append(score)
    alloc = _allocate_integer_budget(scores, budget_total)
    return [max(0, int(x)) for x in alloc]


def select_scale_candidate(
    filter_mode: str,
    filt_results: List[Dict[str, float]],
) -> int:
    accepted_ids = [i for i, r in enumerate(filt_results) if int(r["accepted"]) == 1]
    if not accepted_ids:
        return -1
    if filter_mode == "entropy":
        return min(accepted_ids, key=lambda i: filt_results[i]["entropy"])
    if filter_mode == "confidence":
        return max(accepted_ids, key=lambda i: filt_results[i]["confidence"])
    if filter_mode == "joint":
        return max(
            accepted_ids,
            key=lambda i: (
                filt_results[i]["confidence"] - filt_results[i]["entropy"] - filt_results[i]["disagreement"]
            ),
        )
    return accepted_ids[-1]  # full mode defaults to highest scale

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic data for Chapter-5 experiments.")

    # 数据路径与基础配置
    parser.add_argument("--data_name", type=str, default="DOTA", help="数据集名称（DIOR/DOTA/FGSC）")
    parser.add_argument("--save_dir", type=str, default="./outputs/generated_results", help="合成数据与统计结果保存目录")
    parser.add_argument("--personal_data_path", type=str, default='/picassox/oss-picassox-train-release/segmentation/intern_segmentation/dc1', help="Infinity 个人化/底座模型目录")
    parser.add_argument("--sft_model_path", type=str, default="", help="SFT 微调模型路径，留空则使用默认模型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 类别增广策略
    parser.add_argument(
        "--augment_strategy",
        type=str,
        default="fixed",
        choices=["fixed", "balance", "ratio"],
        help="增广策略：fixed 固定增量；balance 补齐到 target_num；ratio 按原样本比例扩增",
    )
    parser.add_argument("--target_num", type=int, default=500, help="balance 策略下每类目标样本数")
    parser.add_argument("--fixed_add", type=int, default=100, help="fixed 策略下每类新增样本数")
    parser.add_argument("--ratio", type=float, default=1.0, help="ratio 策略下新增比例（ceil(num * ratio)）")

    # 动态预算（按类别熵分配）
    parser.add_argument(
        "--budget_mode",
        type=str,
        default="static",
        choices=["static", "entropy_dynamic"],
        help="预算模式：static 使用基础预算；entropy_dynamic 按类别不确定性动态分配",
    )
    parser.add_argument("--budget_total", type=int, default=-1, help="总预算；<=0 时使用基础预算总和")
    parser.add_argument("--entropy_ckpt", type=str, default="", help="用于估计类别熵的分类器权重路径")
    parser.add_argument(
        "--entropy_mix_imbalance",
        type=float,
        default=0.3,
        help="熵与类不平衡混合系数（score = entropy + coef * imbalance）",
    )
    parser.add_argument(
        "--entropy_temperature",
        type=float,
        default=1.0,
        help="预算分配温度（>1 更平滑，<1 更尖锐）",
    )
    parser.add_argument(
        "--entropy_max_samples_per_class",
        type=int,
        default=200,
        help="每类用于估计熵的最大样本数，<=0 表示不限制",
    )

    # 过滤器（生成后质量筛选）
    parser.add_argument(
        "--filter_mode",
        type=str,
        default="full",
        choices=["full", "confidence", "entropy", "joint"],
        help="过滤模式：full 不过滤；confidence/entropy/joint 按阈值筛选",
    )
    parser.add_argument("--confidence_min", type=float, default=0.50, help="最小置信度阈值")
    parser.add_argument("--entropy_max", type=float, default=0.80, help="最大熵阈值（越小越严格）")
    parser.add_argument("--disagreement_max", type=float, default=0.20, help="最大分歧阈值（集成 KL）")
    parser.add_argument(
        "--classifier_ckpts",
        type=str,
        default="",
        help="过滤器分类器权重，多个用英文逗号分隔",
    )
    parser.add_argument(
        "--classifier_model",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50"],
        help="过滤器分类器骨干网络",
    )

    # 生成采样与输出控制
    parser.add_argument(
        "--use_multiscale_candidates",
        type=int,
        default=0,
        choices=[0, 1],
        help="是否启用单次生成多尺度候选（1 启用，0 仅最终尺度）",
    )
    parser.add_argument(
        "--max_attempt_factor",
        type=float,
        default=5.0,
        help="每类最大尝试倍数：max_attempts = ceil(target_gen * factor)",
    )
    parser.add_argument("--cfg", type=float, default=3.0, help="CFG 引导强度")
    parser.add_argument("--tau", type=float, default=0.5, help="采样温度")
    parser.add_argument("--sampling_per_bits", type=int, default=1, help="VAR 每比特采样次数")
    parser.add_argument("--overwrite", type=int, default=1, choices=[0, 1], help="是否覆盖已存在文件")
    parser.add_argument("--pn", type=str, default="0.06M", help="模型规模标识（与配置中的 pn 对应）")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) Dataset and class metadata
    args.train_path = f"/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/{args.data_name}/train"
    args.test_path = f"/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/{args.data_name}/train"
    train_dataset, test_dataset = get_RS_datasets(args, args.train_path, args.test_path)
    print(f"[INFO] train size={len(train_dataset)}, test size={len(test_dataset)}")

    dataset_name = infer_dataset_name(args.train_path, args.test_path)
    class2label = get_class2label(dataset_name)  # {class_name: label_idx}
    ordered = sorted(class2label.items(), key=lambda kv: kv[1])
    class_names = [x[0] for x in ordered]
    class_indices = [int(x[1]) for x in ordered]

    counts_by_name: Dict[str, int] = {name: 0 for name in train_dataset.classes}
    for _, idx in train_dataset.samples:
        cls_name = train_dataset.classes[idx]
        counts_by_name[cls_name] = counts_by_name.get(cls_name, 0) + 1
    nums_per_class = [int(counts_by_name.get(name, 0)) for name in class_names]

    base_add_nums = set_number_of_samples_per_class(
        nums_per_class=nums_per_class,
        strategy=args.augment_strategy,
        target_num=args.target_num,
        fixed_add=args.fixed_add,
        ratio=args.ratio,
    )

    # 2) Build filter classifier (if needed)
    classifier = None
    if args.filter_mode != "full":
        ckpts = [x.strip() for x in args.classifier_ckpts.split(",") if x.strip()]
        classifier = EnsembleClassifier(
            ckpt_paths=ckpts,
            num_classes=len(class_names),
            device="cuda",
            model_name=args.classifier_model,
        )
    data_filter = GeneratedDataFilter(
        mode=args.filter_mode,
        thresholds=FilterThresholds(
            confidence_min=args.confidence_min,
            entropy_max=args.entropy_max,
            disagreement_max=args.disagreement_max,
        ),
        classifier=classifier,
    )

    # 3) Dynamic budget planning by entropy (Generate What)
    planner_info = {}
    add_nums_per_class = list(base_add_nums)
    if args.budget_mode == "entropy_dynamic":
        entropy_ckpt = args.entropy_ckpt.strip()
        if not entropy_ckpt:
            if not args.classifier_ckpts.strip():
                raise ValueError("entropy_dynamic requires --entropy_ckpt or --classifier_ckpts.")
            entropy_ckpt = args.classifier_ckpts.split(",")[0].strip()
        entropy_classifier = EnsembleClassifier(
            ckpt_paths=[entropy_ckpt],
            num_classes=len(class_names),
            device="cuda",
            model_name=args.classifier_model,
        )
        class_entropy = estimate_class_entropy(
            train_path=args.train_path,
            class2label=class2label,
            classifier=entropy_classifier,
            max_samples_per_class=max(0, int(args.entropy_max_samples_per_class)),
        )
        total_budget = int(sum(base_add_nums)) if args.budget_total <= 0 else int(args.budget_total)
        add_nums_per_class = dynamic_budget_by_entropy(
            base_add_nums=base_add_nums,
            nums_per_class=nums_per_class,
            class_indices=class_indices,
            class_entropy=class_entropy,
            entropy_mix_imbalance=args.entropy_mix_imbalance,
            entropy_temperature=args.entropy_temperature,
            budget_total=total_budget,
        )
        planner_info = {
            "class_entropy": {str(k): float(v) for k, v in class_entropy.items()},
            "base_budget": [int(x) for x in base_add_nums],
            "dynamic_budget": [int(x) for x in add_nums_per_class],
            "budget_total": int(total_budget),
        }

    # 4) Load generative model
    sft_model_path = args.sft_model_path if args.sft_model_path else None
    vae, infinity, text_tokenizer, text_encoder, model_args = get_models(
        personal_data_path=args.personal_data_path,
        sft_models_path=sft_model_path,
        config={"sampling_per_bits": args.sampling_per_bits,
                "pn":'0.06M',
                "vae_type":16,
                "model_type":'infinity_layer12',
                },
    )
    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 1.0))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template][model_args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    os.makedirs(args.save_dir, exist_ok=True)
    all_records: List[Dict[str, object]] = []
    t_run_start = time.time()
    global_attempt = 0
    global_kept = 0

    # 5) Generation loop
    for class_name, class_idx, num_to_generate, num_original in zip(
        class_names, class_indices, add_nums_per_class, nums_per_class
    ):
        if num_to_generate <= 0:
            continue
        prompt = f"A high-resolution satellite top-down view of a {class_name} in a remote sensing image."
        class_dir = os.path.join(args.save_dir, sanitize_filename(class_name))
        os.makedirs(class_dir, exist_ok=True)
        valid_count = 0
        attempts = 0
        max_attempts = max(1, int(math.ceil(num_to_generate * args.max_attempt_factor)))
        print(
            f"[CLASS] {class_name} (label={class_idx}) original={num_original} target_gen={num_to_generate} max_attempts={max_attempts}"
        )

        while valid_count < num_to_generate and attempts < max_attempts:
            attempts += 1
            global_attempt += 1
            seed = random.randint(0, 2**31 - 1)

            gen_t0 = time.time()
            generated = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                prompt,
                g_seed=seed,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[model_args.cfg_insertion_layer],
                vae_type=model_args.vae_type,
                sampling_per_bits=model_args.sampling_per_bits,
                return_all_scales=bool(args.use_multiscale_candidates),
            )
            gen_cost = time.time() - gen_t0

            # Build candidates: final only OR all scales.
            if isinstance(generated, dict):
                scale_imgs = [tensor_hwc_to_uint8_hwc(x) for x in generated["scale_imgs"]]
                candidate_imgs = scale_imgs
            else:
                candidate_imgs = [tensor_hwc_to_uint8_hwc(generated)]

            filt_t0 = time.time()
            filt_results = [data_filter.evaluate(img, target_idx=class_idx) for img in candidate_imgs]
            chosen_scale_idx = select_scale_candidate(args.filter_mode, filt_results)
            filt_cost = time.time() - filt_t0

            chosen = filt_results[chosen_scale_idx] if chosen_scale_idx >= 0 else filt_results[-1]
            accepted = int(chosen_scale_idx >= 0)

            file_rel = ""
            if accepted:
                img_uint8 = candidate_imgs[chosen_scale_idx]
                filename = f"synth_{valid_count:05d}_seed{seed}_s{chosen_scale_idx}.png"
                file_path = os.path.join(class_dir, filename)
                if args.overwrite or not os.path.exists(file_path):
                    # gen_one_img 输出是 BGR，这里转成 RGB 再交给 PIL 保存
                    img_rgb = img_uint8[..., ::-1] if (img_uint8.ndim == 3 and img_uint8.shape[2] == 3) else img_uint8
                    Image.fromarray(img_rgb).save(file_path)
                file_rel = os.path.relpath(file_path, args.save_dir)
                valid_count += 1
                global_kept += 1

            rec = {
                "global_attempt": global_attempt,
                "class_name": class_name,
                "class_idx": class_idx,
                "prompt": prompt,
                "seed": seed,
                "accepted": accepted,
                "reason": chosen["reason"] if accepted else "all_scales_rejected",
                "pred_idx": int(chosen["pred_idx"]),
                "confidence": float(chosen["confidence"]),
                "entropy": float(chosen["entropy"]),
                "disagreement": float(chosen["disagreement"]),
                "chosen_scale_idx": int(chosen_scale_idx),
                "num_scales": int(len(candidate_imgs)),
                "gen_time_s": gen_cost,
                "filter_time_s": filt_cost,
                "total_time_s": gen_cost + filt_cost,
                "file_relpath": file_rel,
            }
            all_records.append(rec)
            print(
                f"  [TRY {attempts}/{max_attempts}] keep={rec['accepted']} scale={rec['chosen_scale_idx']} "
                f"reason={rec['reason']} conf={rec['confidence']:.4f} H={rec['entropy']:.4f} D={rec['disagreement']:.4f} "
                f"kept={valid_count}/{num_to_generate}"
            )

        if attempts >= max_attempts and valid_count < num_to_generate:
            print(f"[WARN] {class_name}: reached max attempts, generated {valid_count}/{num_to_generate}.")

    # 6) Save statistics
    df = pd.DataFrame(all_records)
    records_csv = os.path.join(args.save_dir, "generation_records.csv")
    df.to_csv(records_csv, index=False)

    summary = {
        "filter_mode": args.filter_mode,
        "augment_strategy": args.augment_strategy,
        "budget_mode": args.budget_mode,
        "seed": args.seed,
        "total_attempts": int(global_attempt),
        "total_kept": int(global_kept),
        "pass_rate": float(global_kept / max(global_attempt, 1)),
        "run_time_s": float(time.time() - t_run_start),
        "mean_gen_time_s": float(df["gen_time_s"].mean()) if len(df) else 0.0,
        "mean_filter_time_s": float(df["filter_time_s"].mean()) if len(df) else 0.0,
        "mean_total_time_s": float(df["total_time_s"].mean()) if len(df) else 0.0,
        "thresholds": {
            "confidence_min": args.confidence_min,
            "entropy_max": args.entropy_max,
            "disagreement_max": args.disagreement_max,
        },
        "planner": planner_info,
    }

    if len(df):
        cls_grp = df.groupby(["class_name", "class_idx"], as_index=False).agg(
            attempts=("accepted", "size"),
            kept=("accepted", "sum"),
            mean_conf=("confidence", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_disagreement=("disagreement", "mean"),
            mean_scale=("chosen_scale_idx", "mean"),
            mean_time_s=("total_time_s", "mean"),
        )
        cls_grp["pass_rate"] = cls_grp["kept"] / cls_grp["attempts"].clip(lower=1)
        per_class_csv = os.path.join(args.save_dir, "generation_summary_per_class.csv")
        cls_grp.to_csv(per_class_csv, index=False)
        summary["per_class_csv"] = os.path.relpath(per_class_csv, args.save_dir)
    else:
        summary["per_class_csv"] = ""

    summary_json = os.path.join(args.save_dir, "generation_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] records: {records_csv}")
    print(f"[DONE] summary: {summary_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
