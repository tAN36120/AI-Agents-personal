#!/usr/bin/env python3
"""Map HuggingFace fine-tuned artifacts into NegMerge --results-db layout.

Target layout expected by src/negmerge.py:
  <results_db>/<finetuning_mode>/<model>/
    zeroshot.pt
    zeroshot_accuracies.json
    <any_name_with_'checkpoints'>/<Dataset>Val/finetuned.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

KNOWN_DATASETS = [
    "Cars",
    "SVHN",
    "ImageNet",
    "CIFAR10",
    "CIFAR100",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "DTD",
    "RESISC45",
    "SUN397",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, help="Local folder containing downloaded HF artifacts.")
    parser.add_argument("--results-db", required=True, help="Output root path used by --results-db in negmerge.py")
    parser.add_argument("--finetuning-mode", default="standard", choices=["standard", "linear"])
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlink.")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, do_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if do_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def infer_dataset(path: Path) -> str | None:
    parts = [p.name for p in path.parents] + [path.name]
    for part in parts:
        if part.endswith("Val") and len(part) > 3:
            return part[:-3]
    joined = "/".join(parts)
    for ds in KNOWN_DATASETS:
        if ds in joined:
            return ds
    return None


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    target_root = Path(args.results_db).expanduser().resolve() / args.finetuning_mode / args.model
    target_root.mkdir(parents=True, exist_ok=True)

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    # 1) Required base files
    zeroshot_candidates = sorted(source_root.rglob("zeroshot.pt"))
    linear_zeroshot_candidates = sorted(source_root.rglob("linear_zeroshot.pt"))
    acc_candidates = sorted(source_root.rglob("zeroshot_accuracies.json"))

    if args.finetuning_mode == "linear":
        if not linear_zeroshot_candidates:
            raise FileNotFoundError("linear_zeroshot.pt not found under source root")
        link_or_copy(linear_zeroshot_candidates[0], target_root / "linear_zeroshot.pt", args.copy)
    else:
        if not zeroshot_candidates:
            raise FileNotFoundError("zeroshot.pt not found under source root")
        link_or_copy(zeroshot_candidates[0], target_root / "zeroshot.pt", args.copy)

    if not acc_candidates:
        raise FileNotFoundError("zeroshot_accuracies.json not found under source root")
    link_or_copy(acc_candidates[0], target_root / "zeroshot_accuracies.json", args.copy)

    # 2) Candidate finetuned checkpoints
    finetuned_candidates = sorted(source_root.rglob("finetuned.pt"))
    if not finetuned_candidates:
        raise FileNotFoundError("No finetuned.pt files found under source root")

    mapped = 0
    skipped = 0
    for idx, ckpt in enumerate(finetuned_candidates, start=1):
        dataset = infer_dataset(ckpt)
        if not dataset:
            skipped += 1
            continue
        dst = target_root / f"hf_{idx:03d}_checkpoints" / f"{dataset}Val" / "finetuned.pt"
        link_or_copy(ckpt, dst, args.copy)
        mapped += 1

    print(f"[prepare] source_root={source_root}")
    print(f"[prepare] target_root={target_root}")
    print(f"[prepare] mapped_checkpoints={mapped}, skipped={skipped}")
    print("[prepare] You can now run Experiment 4 with --results-db", Path(args.results_db).expanduser())


if __name__ == "__main__":
    main()
