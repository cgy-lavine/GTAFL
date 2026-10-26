import os
import glob
import argparse
from typing import Tuple, List, Optional

import torch
import torch.nn.functional as F

from options import args_parser as train_args_parser
from Model.Resnet8 import ResNet_cifar
from Dataset.TestAgnostic_CIFAR import TestAgnosticImbalanceCIFAR10DataLoader


def parse_global_sample(v: str) -> bool:
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid global_sample value: {v}")


def get_test_distribution_params(test_distribution: str) -> Tuple[float, bool]:
    # Same mapping as `main.py:create_test_data()`
    distrb = {
        "uniform": (1, False),
        "forward50": (0.02, False),
        "forward25": (0.04, False),
        "forward10": (0.1, False),
        "forward5": (0.2, False),
        "forward2": (0.5, False),
        "backward50": (0.02, True),
        "backward25": (0.04, True),
        "backward10": (0.1, True),
        "backward5": (0.2, True),
        "backward2": (0.5, True),
    }
    if test_distribution not in distrb:
        raise ValueError(f"Unknown test_distribution: {test_distribution}. Valid keys: {list(distrb.keys())}")
    return distrb[test_distribution][0], distrb[test_distribution][1]


def build_test_loader(
    *,
    seed: int,
    path_cifar10: str,
    test_distribution: str,
    batch_size: int,
    num_workers: int = 1,
    imb_type: str = "exp",
) -> torch.utils.data.DataLoader:
    test_imb_factor, reverse = get_test_distribution_params(test_distribution)
    loader = TestAgnosticImbalanceCIFAR10DataLoader(
        seed,
        path_cifar10,
        test_type=test_distribution,
        batch_size=batch_size,
        shuffle=False,
        training=False,
        num_workers=num_workers,
        test_imb_factor=test_imb_factor,
        imb_type=imb_type,
        reverse=reverse,
    )
    return loader.test_set()


def _parse_acc_from_filename(path: str) -> float:
    # Example: model_seed2025_forward_acc0.7422.pth
    base = os.path.basename(path)
    # keep only part after "acc" and before ".pth"
    try:
        acc_str = base.split("acc", 1)[1].rsplit(".pth", 1)[0]
        return float(acc_str)
    except Exception:
        return float("-inf")


def find_best_expert_weight(
    *,
    dir_path: str,
    seed: int,
    expert: str,
) -> str:
    pattern = os.path.join(dir_path, f"model_seed{seed}_{expert}_acc*.pth")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find weights for expert={expert} with pattern:\n  {pattern}\n"
            f"Directory content example:\n  {os.listdir(dir_path)[:20] if os.path.isdir(dir_path) else 'N/A'}"
        )
    # pick highest acc if multiple
    candidates.sort(key=_parse_acc_from_filename, reverse=True)
    return candidates[0]


def load_expert_models(
    *,
    device: str,
    seed: int,
    global_sample: bool,
    test_distribution: str,
    resnet_size: int,
    num_classes: int,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    base_dir = os.path.join(
        "saved_models",
        f"global_sample_{global_sample}",
        f"test_distribution_{test_distribution}",
    )
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Saved model directory not found: {base_dir}")

    forward_path = find_best_expert_weight(dir_path=base_dir, seed=seed, expert="forward")
    uniform_path = find_best_expert_weight(dir_path=base_dir, seed=seed, expert="uniform")
    backward_path = find_best_expert_weight(dir_path=base_dir, seed=seed, expert="backward")

    def build_model():
        return ResNet_cifar(
            resnet_size=resnet_size,
            scaling=4,
            save_activations=False,
            group_norm_num_groups=None,
            freeze_bn=False,
            freeze_bn_affine=False,
            num_classes=num_classes,
        ).to(device)

    model_forward = build_model()
    model_uniform = build_model()
    model_backward = build_model()

    model_forward.load_state_dict(torch.load(forward_path, map_location=device))
    model_uniform.load_state_dict(torch.load(uniform_path, map_location=device))
    model_backward.load_state_dict(torch.load(backward_path, map_location=device))

    model_forward.eval()
    model_uniform.eval()
    model_backward.eval()

    print("Loaded expert weights:")
    print(f"  forward : {forward_path}")
    print(f"  uniform : {uniform_path}")
    print(f"  backward: {backward_path}")

    return model_forward, model_uniform, model_backward


@torch.no_grad()
def evaluate_ensemble(
    *,
    model_forward: torch.nn.Module,
    model_uniform: torch.nn.Module,
    model_backward: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    ensemble_mode: str,
    fixed_weights: List[float],
    normalize_weights: str,
) -> Tuple[float, float, float, float]:
    if len(fixed_weights) != 3:
        raise ValueError("fixed_weights must have length 3: [w_forward, w_uniform, w_backward]")

    total = 0
    correct_ens = 0
    correct_f = 0
    correct_u = 0
    correct_b = 0

    w = torch.tensor(fixed_weights, dtype=torch.float, device=device)
    if normalize_weights == "softmax":
        w = torch.softmax(w, dim=0)
    elif normalize_weights == "sum1":
        w = w / (w.sum().clamp_min(1e-12))
    elif normalize_weights == "none":
        pass
    else:
        raise ValueError("normalize_weights must be one of: softmax, sum1, none")

    for batch in data_loader:
        images, labels = batch
        images = images.to(device)
        # Some datasets may return numpy scalar labels; convert robustly.
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)
        else:
            labels = torch.as_tensor(labels, device=device).long()

        _, logits_f = model_forward(images)
        _, logits_u = model_uniform(images)
        _, logits_b = model_backward(images)

        preds_f = logits_f.argmax(dim=1)
        preds_u = logits_u.argmax(dim=1)
        preds_b = logits_b.argmax(dim=1)
        correct_f += (preds_f == labels).sum().item()
        correct_u += (preds_u == labels).sum().item()
        correct_b += (preds_b == labels).sum().item()

        if ensemble_mode == "fixed":
            combined = w[0] * logits_f + w[1] * logits_u + w[2] * logits_b
        elif ensemble_mode == "confidence":
            # Per-sample adaptive weighting (no amplification):
            # 1) confidence_i(x) = max_j softmax(logits_i(x))[j]
            # 2) weight_i(x) = confidence_i(x) / sum_k confidence_k(x)
            conf_f = F.softmax(logits_f, dim=1).max(dim=1).values  # [B]
            conf_u = F.softmax(logits_u, dim=1).max(dim=1).values
            conf_b = F.softmax(logits_b, dim=1).max(dim=1).values
            conf = torch.stack([conf_f, conf_u, conf_b], dim=0)  # [3,B]
            weights = conf / conf.sum(dim=0, keepdim=True).clamp_min(1e-12)  # [3,B]
            weights = weights.unsqueeze(-1)  # [3,B,1] for broadcasting with logits [B,C]
            combined = weights[0] * logits_f + weights[1] * logits_u + weights[2] * logits_b
        else:
            raise ValueError("ensemble_mode must be one of: fixed, confidence")

        preds = combined.argmax(dim=1)
        correct_ens += (preds == labels).sum().item()
        total += labels.numel()

    acc_f = correct_f / max(total, 1)
    acc_u = correct_u / max(total, 1)
    acc_b = correct_b / max(total, 1)
    acc_ens = correct_ens / max(total, 1)
    return acc_ens, acc_f, acc_u, acc_b


def main():
    # You can edit these defaults directly.
    DEFAULTS = {
        "seed": 2025,
        "global_sample": True,  # -> loads saved_models/global_sample_True/...
        "test_distribution": "forward25",
        "ensemble_mode": "confidence",  # "fixed" or "confidence"
        "fixed_weights": [1.0, 1.0, 1.0],  # [forward, uniform, backward]
        "normalize_weights": "softmax",  # "softmax" / "sum1" / "none"
        "batch_size_test": 128,
    }

    # Optional CLI overrides (keep arguments minimal so it's easy to run experiments).
    parser = argparse.ArgumentParser("Adaptive ensemble evaluation")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--global_sample", type=str, default=str(DEFAULTS["global_sample"]))
    parser.add_argument("--test_distribution", type=str, default=DEFAULTS["test_distribution"])
    parser.add_argument("--ensemble_mode", type=str, default=DEFAULTS["ensemble_mode"], choices=["fixed", "confidence"])
    parser.add_argument("--w_forward", type=float, default=DEFAULTS["fixed_weights"][0])
    parser.add_argument("--w_uniform", type=float, default=DEFAULTS["fixed_weights"][1])
    parser.add_argument("--w_backward", type=float, default=DEFAULTS["fixed_weights"][2])
    parser.add_argument("--normalize_weights", type=str, default=DEFAULTS["normalize_weights"], choices=["softmax", "sum1", "none"])
    parser.add_argument("--batch_size_test", type=int, default=DEFAULTS["batch_size_test"])
    parser.add_argument("--resnet_size", type=int, default=8)
    parser.add_argument("--imb_type", type=str, default="exp")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Reuse training defaults for paths so you don't have to manually set them.
    train_args = train_args_parser()
    parser.add_argument("--path_cifar10", type=str, default=train_args.path_cifar10)

    args = parser.parse_args()

    global_sample_bool = parse_global_sample(args.global_sample)

    data_loader = build_test_loader(
        seed=args.seed,
        path_cifar10=args.path_cifar10,
        test_distribution=args.test_distribution,
        batch_size=args.batch_size_test,
        num_workers=1,
        imb_type=args.imb_type,
    )

    model_forward, model_uniform, model_backward = load_expert_models(
        device=args.device,
        seed=args.seed,
        global_sample=global_sample_bool,
        test_distribution=args.test_distribution,
        resnet_size=args.resnet_size,
        num_classes=10,
    )

    acc = evaluate_ensemble(
        model_forward=model_forward,
        model_uniform=model_uniform,
        model_backward=model_backward,
        data_loader=data_loader,
        device=args.device,
        ensemble_mode=args.ensemble_mode,
        fixed_weights=[args.w_forward, args.w_uniform, args.w_backward],
        normalize_weights=args.normalize_weights,
    )

    print(f"Ensemble mode: {args.ensemble_mode}")
    print(f"Test distribution: {args.test_distribution}")
    print(f"global_sample: {global_sample_bool}, seed: {args.seed}")
    acc_ens, acc_f, acc_u, acc_b = acc
    print(f"Expert acc (forward/uniform/backward): {acc_f*100:.4f}% / {acc_u*100:.4f}% / {acc_b*100:.4f}%")
    print(f"Ensemble accuracy: {acc_ens*100:.4f}%")

    # Logging
    os.makedirs("main_result/ensemble_logs", exist_ok=True)
    w_tag = f"wf{args.w_forward:.3f}-wu{args.w_uniform:.3f}-wb{args.w_backward:.3f}"
    log_filename = (
        f"ensemble_{args.test_distribution}_"
        f"mode{args.ensemble_mode}_"
        f"global{global_sample_bool}_seed{args.seed}_"
        f"{w_tag}.txt"
    )
    log_path = os.path.join("main_result/ensemble_logs", log_filename)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("Adaptive ensemble evaluation\n")
        f.write(f"test_distribution: {args.test_distribution}\n")
        f.write(f"ensemble_mode: {args.ensemble_mode}\n")
        f.write(f"global_sample: {global_sample_bool}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"fixed_weights(raw): forward={args.w_forward}, uniform={args.w_uniform}, backward={args.w_backward}\n")
        f.write(f"normalize_weights: {args.normalize_weights}\n")
        f.write("\n")
        f.write(f"Expert accuracy:\n")
        f.write(f"  forward : {acc_f:.6f}\n")
        f.write(f"  uniform : {acc_u:.6f}\n")
        f.write(f"  backward : {acc_b:.6f}\n")
        f.write("\n")
        f.write(f"Ensemble accuracy: {acc_ens:.6f}\n")
        f.write("\n")
        f.write("Note: Expert weights are loaded from saved_models/*/test_distribution_* using filename acc*.pth (best acc).\n")

    print(f"Saved log to: {log_path}")


if __name__ == "__main__":
    main()

