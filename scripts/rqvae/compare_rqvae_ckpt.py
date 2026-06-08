#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import torch


DATASET_MAP = {
    "beauty": "beauty",
    "instruments": "instruments",
    "yelp": "yelp",
}

DEFAULT_BASELINE_TAGS = {
    "beauty": "beauty_strong_sinkhorn",
    "instruments": "instruments_strong_sinkhorn",
    "yelp": "yelp_strong_sinkhorn",
}

DEFAULT_BASELINE_STAMPS = {
    "beauty": "Nov-03-2025_16-13-56",
    "instruments": "Dec-04-2025_14-48-43",
    "yelp": "Dec-07-2025_20-22-35",
}

TARGETS = {
    "beauty": {
        "epochs": 10000,
        "batch_size": 1024,
        "beta": 0.25,
        "sk_iters": 50,
        "sk_epsilons": [0.003, 0.003, 0.003],
        "num_emb_list": [256, 256, 256],
        "layers": [2048, 1024, 512],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "e_dim": 256,
        "vq_type": "vq",
        "dist": "l2",
    },
    "instruments": {
        "epochs": 10000,
        "batch_size": 2048,
        "beta": 0.25,
        "sk_iters": 50,
        "sk_epsilons": [0.003, 0.003, 0.003],
        "num_emb_list": [256, 256, 256],
        "layers": [2048, 1024, 512],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "e_dim": 256,
        "vq_type": "vq",
        "dist": "l2",
    },
    "yelp": {
        "epochs": 10000,
        "batch_size": 4096,
        "beta": 0.5,
        "sk_iters": 50,
        "sk_epsilons": [0.003, 0.003, 0.003],
        "num_emb_list": [256, 256, 256],
        "layers": [2048, 1024, 512],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "e_dim": 256,
        "vq_type": "vq",
        "dist": "l2",
    },
}


def _parse_pairs(items: List[str], label: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{label} expects dataset=value items, got '{item}'")
        ds, val = item.split("=", 1)
        ds = ds.strip()
        val = val.strip()
        if not ds or not val:
            raise ValueError(f"Invalid {label} entry '{item}'")
        parsed[ds] = val
    return parsed


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_args(x):
    if hasattr(x, "__dict__"):
        x = dict(x.__dict__)
    elif isinstance(x, dict):
        x = dict(x)
    else:
        return None
    # Keep comparison focus stable across runs.
    x.pop("device", None)
    return x


def _close_or_equal(a, b, abs_tol=1e-8):
    try:
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(abs(float(x) - float(y)) <= abs_tol for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= abs_tol
    except Exception:
        return a == b


def _load(path: Path):
    return torch.load(str(path), map_location='cpu', weights_only=False)


def _pick_baseline_ckpt(root: Path, dataset: str, baseline_tag: str, baseline_stamp: str) -> Path:
    return root / baseline_tag / baseline_stamp / "best_collision_model.pth"


def _get_ckpt_summary(path: Path) -> Tuple[Dict, float, int, int, str]:
    ckpt = _load(path)
    args = ckpt.get('args', {})
    if args is not None and not isinstance(args, dict):
        args = _normalize_args(args) or {}
    args = args or {}
    best_collision = float(ckpt.get('best_collision_rate', float('nan')))
    epoch = int(ckpt.get('epoch', -1)) if ckpt.get('epoch', None) is not None else -1
    state_dict = ckpt.get('state_dict', {}) or {}
    total_params = 0
    for v in state_dict.values():
        if hasattr(v, 'numel'):
            total_params += int(v.numel())
    file_hash = _file_sha256(path)
    return args, best_collision, epoch, total_params, file_hash


def _check_dataset(
    dataset: str,
    target_root: Path,
    baseline_root: Path,
    strict: bool,
    expect_hash: bool,
    base_tag: Dict[str, str],
    base_stamp: Dict[str, str],
) -> bool:
    if dataset not in DATASET_MAP:
        print(f"[FAIL] Unknown dataset '{dataset}'")
        return False

    target_ckpt = target_root / DATASET_MAP[dataset] / "best_collision_model.pth"
    if not target_ckpt.exists():
        print(f"[FAIL] {dataset}: missing candidate -> {target_ckpt}")
        return False

    baseline_tag = base_tag.get(dataset, DEFAULT_BASELINE_TAGS[dataset])
    baseline_stamp = base_stamp.get(dataset, DEFAULT_BASELINE_STAMPS[dataset])
    baseline_ckpt = _pick_baseline_ckpt(baseline_root, dataset, baseline_tag, baseline_stamp)
    if not baseline_ckpt.exists():
        print(f"[FAIL] {dataset}: missing baseline -> {baseline_ckpt}")
        return False

    cand_args, cand_collision, cand_epoch, cand_params, cand_sha = _get_ckpt_summary(target_ckpt)
    base_args, base_collision, base_epoch, base_params, base_sha = _get_ckpt_summary(baseline_ckpt)

    print(f"[{dataset}] candidate={target_ckpt.name}, baseline={baseline_ckpt}")
    print(f"  epoch: cand={cand_epoch}, base={base_epoch}")
    print(f"  best_collision: cand={cand_collision:.12f}, base={base_collision:.12f}")
    print(f"  params: cand={cand_params}, base={base_params}")

    ok = True
    if expect_hash:
        print(f"  sha256: cand={cand_sha}")
        print(f"  sha256: base={base_sha}")
        if cand_sha != base_sha:
            print("  [DIFF] file hashes differ")
            ok = False
        else:
            print("  [OK] file hashes match")

    target_cfg = TARGETS[dataset]
    for k, expected in target_cfg.items():
        val = cand_args.get(k)
        if val is None:
            print(f"  [MISS] candidate missing args.{k}")
            ok = False
            continue
        if isinstance(expected, (list, tuple)):
            same = _close_or_equal(val, expected, abs_tol=0.0)
        else:
            same = _close_or_equal(val, expected, abs_tol=1e-12 if isinstance(expected, float) else 0.0)
        if not same:
            print(f"  [DIFF] args.{k}: cand={val}, target={expected}")
            ok = False
        else:
            print(f"  [OK] args.{k}: {val}")

    if strict:
        if cand_collision != base_collision:
            print(f"  [DIFF] strict mode: best_collision differs exactly ({cand_collision} != {base_collision})")
            ok = False
        else:
            print("  [OK] strict mode: best_collision exact match")
        if cand_epoch != base_epoch:
            print(f"  [DIFF] strict mode: epoch differs ({cand_epoch} != {base_epoch})")
            ok = False
    else:
        delta = abs(cand_collision - base_collision)
        if delta > 1e-6:
            print(f"  [WARN] collision deviates by {delta:.8f}")
        else:
            print("  [OK] collision close enough (<1e-6)")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Compare reproduced rq-vae ckpts against baseline originals.")
    parser.add_argument('--ckpt_root', default='./rqvae_ckpt', help='Candidate root (default: ./rqvae_ckpt)')
    parser.add_argument(
        '--baseline_root',
        default='/data/junch/ETEGRec_ONE_Stage/RQVAE/rqvae_ckpt',
        help='Baseline checkpoint root',
    )
    parser.add_argument('--expect_hash', action='store_true', help='Require candidate and baseline file hash to match')
    parser.add_argument('--strict', action='store_true', help='Require exact collision and epoch equality')
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=['beauty', 'instruments', 'yelp'],
        help='Datasets to compare',
    )
    parser.add_argument('--base_tag', nargs='*', help='Override baseline tag as dataset=tag')
    parser.add_argument('--base_stamp', nargs='*', help='Override baseline timestamp as dataset=stamp')

    args = parser.parse_args()

    try:
        base_tag = _parse_pairs(args.base_tag or [], '--base_tag')
        base_stamp = _parse_pairs(args.base_stamp or [], '--base_stamp')
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 1

    target_root = Path(args.ckpt_root)
    baseline_root = Path(args.baseline_root)

    if not target_root.exists():
        print(f"[FAIL] Candidate root does not exist: {target_root}")
        return 1
    if not baseline_root.exists():
        print(f"[FAIL] Baseline root does not exist: {baseline_root}")
        return 1

    invalid = [ds for ds in args.datasets if ds not in DATASET_MAP]
    if invalid:
        print(f"[FAIL] Unsupported datasets: {', '.join(invalid)}")
        return 1

    ok_all = True
    for ds in args.datasets:
        ok = _check_dataset(
            ds,
            target_root=target_root,
            baseline_root=baseline_root,
            strict=args.strict,
            expect_hash=args.expect_hash,
            base_tag=base_tag,
            base_stamp=base_stamp,
        )
        ok_all = ok_all and ok

    if ok_all:
        print('\n[PASS] All datasets passed comparison checks.')
        return 0
    print('\n[FAIL] One or more datasets differ from baseline.')
    return 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
