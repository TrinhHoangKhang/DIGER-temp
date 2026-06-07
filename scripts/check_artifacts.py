#!/usr/bin/env python3
from pathlib import Path
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "beauty": {
        "embedding": "Beauty.emb-llama.npy",
        "shape": (12101, 4096),
        "map_items": 12102,
        "splits": {"train": 131413, "valid": 22363, "test": 22363},
    },
    "instruments": {
        "embedding": "Instruments.emb-llama.npy",
        "shape": (9922, 4096),
        "map_items": 9923,
        "splits": {"train": 131837, "valid": 24772, "test": 24772},
    },
    "yelp": {
        "embedding": "Yelp.emb-llama.npy",
        "shape": (20033, 4096),
        "map_items": 20034,
        "splits": {"train": 225061, "valid": 30431, "test": 30431},
    },
}


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def check_file(path: Path, errors: list[str]) -> bool:
    if not path.exists():
        errors.append(f"missing file: {path.relative_to(ROOT)}")
        return False
    if is_lfs_pointer(path):
        errors.append(f"LFS pointer was not pulled: {path.relative_to(ROOT)}")
        return False
    return True


def main() -> int:
    errors: list[str] = []

    for dataset, expected in EXPECTED.items():
        dataset_dir = ROOT / "dataset" / dataset
        map_path = dataset_dir / f"{dataset}.emb_map.json"
        emb_path = dataset_dir / expected["embedding"]
        ckpt_path = ROOT / "rqvae_ckpt" / dataset / "best_collision_model.pth"

        if check_file(map_path, errors):
            with map_path.open() as handle:
                map_items = len(json.load(handle))
            if map_items != expected["map_items"]:
                errors.append(f"{dataset}: map has {map_items} items, expected {expected['map_items']}")

        if check_file(emb_path, errors):
            emb = np.load(emb_path, mmap_mode="r")
            if tuple(emb.shape) != expected["shape"]:
                errors.append(f"{dataset}: embedding shape is {tuple(emb.shape)}, expected {expected['shape']}")

        if check_file(ckpt_path, errors) and ckpt_path.stat().st_size < 100_000_000:
            errors.append(f"{dataset}: checkpoint is unexpectedly small: {ckpt_path.stat().st_size} bytes")

        for split, expected_lines in expected["splits"].items():
            split_path = dataset_dir / f"{dataset}.{split}.jsonl"
            if check_file(split_path, errors):
                actual_lines = count_lines(split_path)
                if actual_lines != expected_lines:
                    errors.append(f"{dataset}.{split}: {actual_lines} rows, expected {expected_lines}")

        print(
            f"{dataset}: ok "
            f"items={expected['map_items']} "
            f"embedding={expected['shape']} "
            f"splits={expected['splits']}"
        )

    if errors:
        print("\nArtifact check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("\nAll released datasets, embeddings, and RQ-VAE checkpoints are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
