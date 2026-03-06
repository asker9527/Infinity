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
        self.num_classes = num_classes
        self.models = []
        for item in ckpt_paths:
            backbone, ckpt_path = self._parse_expert_spec(item, default_model=model_name)
            self.models.append(self._load_one(ckpt_path, num_classes, backbone))
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _parse_expert_spec(self, spec: str, default_model: str) -> Tuple[str, str]:
        s = spec.strip()
        # 支持: "resnet18:/a/b.pth"
        if ":" in s:
            left, right = s.split(":", 1)
            left = left.strip().lower()
            right = right.strip()
            if left in {"resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"} and right:
                return left, right
        # 兼容旧格式: "/a/b.pth"
        return default_model, s

    def _build_model(self, model_name: str, num_classes: int) -> torch.nn.Module:
        if model_name == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            return model
        if model_name == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            return model
        if model_name == "mobilenet_v2":
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
            return model
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(in_features, num_classes)
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

        probs_tk = self.classifier.predict_probs(image_hwc_uint8)  
        p_bar = probs_tk.mean(dim=0)
        pred_idx = int(torch.argmax(p_bar).item())

        eps = 1e-8
        logk = math.log(p_bar.numel())

        # 改动点：平均 confidence（专家级 top1 的平均）
        confidence = float(torch.max(probs_tk, dim=1).values.mean().item())

        # 改动点：平均 entropy（专家级 entropy 的平均）
        entropy_t = (-(probs_tk * torch.log(probs_tk + eps)).sum(dim=1) / logk)
        entropy = float(entropy_t.mean().item())

        # 保留原有分歧定义
        kls = []
        for t in range(probs_tk.shape[0]):
            p_t = probs_tk[t]
            kl = torch.sum(p_t * (torch.log(p_t + eps) - torch.log(p_bar + eps)))
            kls.append(kl)
        disagreement = float((torch.stack(kls).mean() / logk).item())

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
        # 改动点：按专家 entropy 求平均
        k = probs_tk.shape[1]
        entropy = float((-(probs_tk * torch.log(probs_tk + eps)).sum(dim=1) / math.log(k)).mean().item())
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

def is_topk_mode(filter_mode: str) -> bool:
    return filter_mode in {"topk_entropy", "topk_confidence", "topk_joint"}


def topk_score(filter_mode: str, rec: Dict[str, float]) -> float:
    if filter_mode == "topk_entropy":
        return -float(rec["entropy"])  # entropy 越小越好
    if filter_mode == "topk_confidence":
        return float(rec["confidence"])  # confidence 越大越好
    if filter_mode == "topk_joint":
        return float(rec["confidence"] - rec["entropy"] - rec["disagreement"])  # 保持原 joint 分数
    raise ValueError(f"Unsupported top-k mode: {filter_mode}")


def select_scale_candidate_topk(
    filter_mode: str,
    filt_results: List[Dict[str, float]],
    target_idx: int,
) -> int:
    # 先按类别匹配优先，再按分数
    return max(
        range(len(filt_results)),
        key=lambda i: (
            int(filt_results[i]["pred_idx"]) == int(target_idx),
            topk_score(filter_mode, filt_results[i]),
        ),
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic data for Chapter-5 experiments.")

    # 数据路径与基础配置
    parser.add_argument("--data_name", type=str, default="DOTA", help="数据集名称（DIOR/DOTA/FGSC）")
    parser.add_argument("--save_dir", type=str, default="./outputs/Generated_Results", help="合成数据与统计结果保存目录")
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
        choices=["full", "confidence", "entropy", "joint", "topk_entropy", "topk_confidence", "topk_joint"],
        help="过滤模式：full/阈值模式/top-k 模式",
    )
    parser.add_argument("--confidence_min", type=float, default=0.50, help="最小置信度阈值")
    parser.add_argument("--entropy_max", type=float, default=0.80, help="最大熵阈值（越小越严格）")
    parser.add_argument("--disagreement_max", type=float, default=0.20, help="最大分歧阈值（集成 KL）")
    parser.add_argument(
        "--classifier_ckpts",
        type=str,
        default="resnet18:/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/models/Mulit_Classifer/FGSC/resnet18_bestmodel.pth",
        help="过滤器分类器权重，多个用逗号分隔；支持 backbone:path（如 resnet18:/a.pth,mobilenet_v2:/b.pth）",
    )
    parser.add_argument(
        "--classifier_model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"],
        help="默认骨干（当 classifier_ckpts 未写 backbone 前缀时使用）",
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
    parser.add_argument("--overwrite", type=int, default=0, choices=[0, 1], help="是否覆盖已存在文件")
    parser.add_argument("--pn", type=str, default="0.06M", help="模型规模标识（与配置中的 pn 对应）")

    # 参考图引导（简化版）
    parser.add_argument("--use_reference_image", type=int, default=1, choices=[0, 1], help="是否启用参考图引导")
    parser.add_argument(
        "--reference_scope",
        type=str,
        default="all",
        choices=["class", "all"],
        help="参考图范围：class=同类别随机，all=全类别随机",
    )
    parser.add_argument("--reference_gt_leak", type=int, default=2, help="传入 gen_one_img 的 gt_leak")

    return parser.parse_args()

def pick_reference_image_from_train_dataset(
    train_dataset,
    class_name: str,
    scope: str = "class",
) -> str:
    if not getattr(train_dataset, "samples", None):
        return ""

    if scope == "all":
        return random.choice(train_dataset.samples)[0]

    # scope == "class": 直接用 train_dataset 的类名做一致匹配
    candidates = [p for p, ds_idx in train_dataset.samples if train_dataset.classes[ds_idx] == class_name]
    if not candidates:
        return random.choice(train_dataset.samples)[0]
    return random.choice(candidates)


@torch.no_grad()
def extract_reference_gt_indices(
    ref_img_path: str,
    vae,
    scale_schedule,
    device: torch.device,
) -> List[torch.Tensor]:
    lanczos = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    img = Image.open(ref_img_path).convert("RGB").resize((256, 256), resample=lanczos)
    x = transforms.ToTensor()(img).unsqueeze(0).to(device, non_blocking=True)  # [1,3,256,256], [0,1]
    x = x.add(x).add_(-1)  # [-1,1]

    raw_features, _, _ = vae.encode_for_raw_features(x, scale_schedule=scale_schedule)
    codes_out = raw_features.unsqueeze(2) if raw_features.dim() == 4 else raw_features
    cum_var_input = torch.zeros_like(codes_out)

    gt_all_bit_indices: List[torch.Tensor] = []
    for si, _ in enumerate(scale_schedule):
        residual = codes_out - cum_var_input
        if si != len(scale_schedule) - 1:
            residual = F.interpolate(residual, size=scale_schedule[si], mode=vae.quantizer.z_interplote_down).contiguous()

        quantized, _, bit_indices, _ = vae.quantizer.lfq(residual)
        gt_all_bit_indices.append(bit_indices)

        cum_var_input = cum_var_input + F.interpolate(
            quantized, size=scale_schedule[-1], mode=vae.quantizer.z_interplote_up
        ).contiguous()

    return gt_all_bit_indices

def count_existing_synth_images(class_dir: str) -> int:
    if not os.path.isdir(class_dir):
        return 0
    # 只统计最终文件名，不统计 cand_*
    return sum(
        1
        for fn in os.listdir(class_dir)
        if fn.startswith("synth_") and fn.endswith(".png")
    )

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
    nums_per_class = [int(counts_by_name.get(str(name), 0)) for name in class_indices]
    base_add_nums = set_number_of_samples_per_class(
        nums_per_class=nums_per_class,
        strategy=args.augment_strategy,
        target_num=args.target_num,
        fixed_add=args.fixed_add,
        ratio=args.ratio,
    )
    print(f"[INFO] base_add_nums={base_add_nums}"*10)
    # 2) Build filter classifier (if needed)
    # Top-K 模式下，评估仍复用基础指标模式
    eval_filter_mode = args.filter_mode
    if is_topk_mode(args.filter_mode):
        eval_filter_mode = args.filter_mode.replace("topk_", "")

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
        mode=eval_filter_mode,  # 修复：不要直接传 topk_*
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

    # 不再构建任何参考图缓存/池，直接使用 train_dataset 即时采样
    try:
        vae_device = next(vae.parameters()).device
    except StopIteration:
        vae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        class_dir = os.path.join(args.save_dir, sanitize_filename(str(class_idx)))
        os.makedirs(class_dir, exist_ok=True)
        valid_count = 0
        attempts = 0
        topk_mode = is_topk_mode(args.filter_mode)
        class_rec_indices: List[int] = []

        # 新增：overwrite=0 时的快速跳过/续跑
        existing_synth = count_existing_synth_images(class_dir)
        if not bool(args.overwrite):
            if existing_synth >= num_to_generate:
                print(f"[SKIP] {class_name}: existing synth={existing_synth} >= target={num_to_generate}")
                continue
            if topk_mode and existing_synth > 0:
                print(
                    f"[SKIP] {class_name}: top-k mode with overwrite=0 and existing synth={existing_synth}. "
                    f"Please clean class dir or set --overwrite 1."
                )
                continue
            if (not topk_mode) and existing_synth > 0:
                valid_count = existing_synth
                print(f"[RESUME] {class_name}: existing synth={existing_synth}, continue to target={num_to_generate}")

        if topk_mode:
            max_attempts = max(1, int(math.ceil(num_to_generate * 3.0)))  # 候选倍数固定为 3
        else:
            max_attempts = max(1, int(math.ceil(num_to_generate * args.max_attempt_factor)))

        print(
            f"[CLASS] {class_name} (label={class_idx}) original={num_original} target_gen={num_to_generate} max_attempts={max_attempts}"
        )

        while (attempts < max_attempts) if topk_mode else (valid_count < num_to_generate and attempts < max_attempts):
            attempts += 1
            global_attempt += 1
            seed = random.randint(0, 2**31 - 1)

            # 参考图：直接从 train_dataset 按 scope 即时随机
            ref_img_path = ""
            ref_gt_ls_Bl = None
            if bool(args.use_reference_image):
                ref_img_path = pick_reference_image_from_train_dataset(
                    train_dataset=train_dataset,
                    class_name=class_name,
                    scope=args.reference_scope,
                )
                if ref_img_path:
                    ref_gt_ls_Bl = extract_reference_gt_indices(
                        ref_img_path=ref_img_path,
                        vae=vae,
                        scale_schedule=scale_schedule,
                        device=vae_device,
                    )

            gen_kwargs = dict(
                g_seed=seed,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[model_args.cfg_insertion_layer],
                vae_type=model_args.vae_type,
                sampling_per_bits=model_args.sampling_per_bits,
                return_all_scales=bool(args.use_multiscale_candidates),
            )
            if ref_gt_ls_Bl is not None:
                gen_kwargs.update(gt_leak=float(args.reference_gt_leak), gt_ls_Bl=ref_gt_ls_Bl)
            gen_time_start = time.time()
            generated = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                prompt,
                **gen_kwargs,
            )
            gen_time_end = time.time()
            gen_cost = gen_time_end - gen_time_start

            # Build candidates: final only OR all scales.
            if isinstance(generated, dict):
                scale_imgs = [tensor_hwc_to_uint8_hwc(x) for x in generated["scale_imgs"]]
                candidate_imgs = scale_imgs
            else:
                candidate_imgs = [tensor_hwc_to_uint8_hwc(generated)]

            filt_t0 = time.time()
            filt_results = [data_filter.evaluate(img, target_idx=class_idx) for img in candidate_imgs]
            filt_cost = time.time() - filt_t0

            if topk_mode:
                chosen_scale_idx = select_scale_candidate_topk(args.filter_mode, filt_results, target_idx=class_idx)
                chosen = filt_results[chosen_scale_idx]
                accepted = 0  # 先不通过，类内结束后统一 Top-K 决定
                score = topk_score(args.filter_mode, chosen)
            else:
                chosen_scale_idx = select_scale_candidate(args.filter_mode, filt_results)
                chosen = filt_results[chosen_scale_idx] if chosen_scale_idx >= 0 else filt_results[-1]
                accepted = int(chosen_scale_idx >= 0)
                score = 0.0

            file_rel = ""
            if topk_mode:
                img_uint8 = candidate_imgs[chosen_scale_idx]
                filename = f"cand_{attempts:05d}_seed{seed}_s{chosen_scale_idx}.png"
                file_path = os.path.join(class_dir, filename)
                if args.overwrite or not os.path.exists(file_path):
                    img_rgb = img_uint8[..., ::-1] if (img_uint8.ndim == 3 and img_uint8.shape[2] == 3) else img_uint8
                    Image.fromarray(img_rgb).save(file_path)
                file_rel = os.path.relpath(file_path, args.save_dir)
            elif accepted:
                img_uint8 = candidate_imgs[chosen_scale_idx]
                filename = f"synth_{valid_count:05d}_seed{seed}_s{chosen_scale_idx}.png"
                file_path = os.path.join(class_dir, filename)
                if args.overwrite or not os.path.exists(file_path):
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
                "reason": chosen["reason"] if (accepted and not topk_mode) else ("topk_candidate" if topk_mode else "all_scales_rejected"),
                "pred_idx": int(chosen["pred_idx"]),
                "confidence": float(chosen["confidence"]),
                "entropy": float(chosen["entropy"]),
                "disagreement": float(chosen["disagreement"]),
                "topk_score": float(score),
                "chosen_scale_idx": int(chosen_scale_idx),
                "num_scales": int(len(candidate_imgs)),
                "gen_time_s": gen_cost,
                "filter_time_s": filt_cost,
                "total_time_s": gen_cost + filt_cost,
                "file_relpath": file_rel,
                "ref_used": int(ref_gt_ls_Bl is not None),
                "ref_scope": args.reference_scope if ref_gt_ls_Bl is not None else "",
                "ref_image_relpath": os.path.relpath(ref_img_path, args.train_path) if ref_gt_ls_Bl is not None else "",
            }
            all_records.append(rec)
            if topk_mode:
                class_rec_indices.append(len(all_records) - 1)

            print(
                f"  [TRY {attempts}/{max_attempts}] keep={rec['accepted']} scale={rec['chosen_scale_idx']} "
                f"reason={rec['reason']} conf={rec['confidence']:.4f} H={rec['entropy']:.4f} D={rec['disagreement']:.4f} "
                f"ref={rec['ref_used']} kept={valid_count}/{num_to_generate}"
            )

        if topk_mode:
            ranked = sorted(
                class_rec_indices,
                key=lambda idx: (
                    int(all_records[idx]["pred_idx"]) == int(class_idx),
                    float(all_records[idx]["topk_score"]),
                ),
                reverse=True,
            )
            keep_set = set(ranked[:num_to_generate])
            new_valid = 0
            for idx in class_rec_indices:
                rec = all_records[idx]
                old_path = os.path.join(args.save_dir, rec["file_relpath"]) if rec["file_relpath"] else ""
                if idx in keep_set:
                    new_name = f"synth_{new_valid:05d}_seed{rec['seed']}_s{rec['chosen_scale_idx']}.png"
                    new_path = os.path.join(class_dir, new_name)
                    if old_path and os.path.exists(old_path):
                        os.replace(old_path, new_path)
                    rec["accepted"] = 1
                    rec["reason"] = "topk_keep"
                    rec["file_relpath"] = os.path.relpath(new_path, args.save_dir)
                    new_valid += 1
                    global_kept += 1
                else:
                    rec["accepted"] = 0
                    rec["reason"] = "topk_drop"
                    if old_path and os.path.exists(old_path):
                        os.remove(old_path)
                    rec["file_relpath"] = ""
            valid_count = new_valid
            if valid_count < num_to_generate:
                print(f"[WARN] {class_name}: top-k kept {valid_count}/{num_to_generate}.")

        if (not topk_mode) and attempts >= max_attempts and valid_count < num_to_generate:
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
        "reference": {
            "use_reference_image": int(args.use_reference_image),
            "reference_scope": args.reference_scope,
            "reference_gt_leak": float(args.reference_gt_leak),
        },
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
