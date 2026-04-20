# NegMerge
# Copyright (c) 2025-present NAVER Cloud Corp.
# MIT license
# Referenced from: https://github.com/gortizji/tangent_task_arithmetic/blob/main/src/eval_task_negation.py

import json
import os
import re

import torch

from src.args import parse_arguments
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from utils import find_optimal_coef

args = parse_arguments()
merge_datasets = args.merge_datasets

args.save = f"{args.results_db}/{args.finetuning_mode}/{args.model}"

with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

control_dataset = "ImageNet"
negation_accuracies = {}

for dataset in merge_datasets:
    pretrained_checkpoint = (
        f"{args.save}/linear_zeroshot.pt"
        if args.finetuning_mode == "linear"
        else f"{args.save}/zeroshot.pt"
    )

    checkpoint_paths = []
    for dir_name in os.listdir(args.save):
        dir_path = os.path.join(args.save, dir_name)
        if not (os.path.isdir(dir_path) and "checkpoints" in dir_name):
            continue
        dataset_val_dir = os.path.join(dir_path, f"{dataset}Val")
        finetuned_ckpt = os.path.join(dataset_val_dir, "finetuned.pt")
        if os.path.isfile(finetuned_ckpt):
            checkpoint_paths.append(finetuned_ckpt)

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No candidate checkpoints found for dataset '{dataset}' under {args.save}. "
            "Expected format: *checkpoints*/<Dataset>Val/finetuned.pt"
        )

    for idx, checkpoint_path in enumerate(sorted(checkpoint_paths)):
        task_vector = (
            LinearizedTaskVector(pretrained_checkpoint, checkpoint_path)
            if args.finetuning_mode == "linear"
            else NonLinearTaskVector(pretrained_checkpoint, checkpoint_path)
        )

        if idx == 0:
            merged_vector = {k: torch.zeros_like(v) for k, v in task_vector.vector.items()}
            mask = {k: torch.zeros_like(v) for k, v in task_vector.vector.items()}

        for key in task_vector.vector.keys():
            merged_vector[key] += task_vector.vector[key]
            mask[key] += torch.sign(task_vector.vector[key])

    for key in task_vector.vector.keys():
        # 1. Standard NegMerge consensus logic
        consistency_mask = torch.abs(mask[key]) == len(checkpoint_paths)
        merged_val = torch.where(
            consistency_mask,
            merged_vector[key] / len(checkpoint_paths),
            torch.zeros_like(merged_vector[key]),
        )

        # 2. Experiment 4: layer-wise zeroing logic
        if args.layer_wise_zeroing:
            is_shallow = False

            # Parameter belongs to a transformer block
            match = re.search(r"resblocks\.(\d+)\.", key)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx <= args.shallow_threshold:
                    is_shallow = True
            else:
                # Stem/embeddings (before block 0)
                stem_keywords = ["conv1", "class_embedding", "positional_embedding", "ln_pre"]
                if any(k in key for k in stem_keywords) and args.shallow_threshold >= 0:
                    is_shallow = True

            if is_shallow:
                merged_val = torch.zeros_like(merged_val)

        # Reassign computed value back to the task vector
        task_vector.vector[key] = merged_val

    args.eval_datasets = [dataset + "Val"]
    args.control_dataset = control_dataset + "Val"
    val_metrics = evaluate_task_vector(
        -task_vector,
        pretrained_checkpoint,
        args,
    )

    optimal_coef = find_optimal_coef(
        val_metrics,
        metric=f"{dataset}Val:top1",
        minimize=True,
        control_metric=f"{control_dataset}Val:top1",
        control_metric_threshold=0.95 * pretrained_accuracies[control_dataset + "Val"],
    )

    # Evaluate on the test set with the optimal coefficient.
    args.eval_datasets = [dataset]
    args.control_dataset = control_dataset
    test_metrics = evaluate_task_vector_at_coef(
        -task_vector,
        pretrained_checkpoint,
        args,
        optimal_coef,
    )

    print("=" * 100)
    print(f"[{dataset}] #candidates={len(checkpoint_paths)}  threshold={args.shallow_threshold}")
    print(f"[{dataset}] Test forget accuracy: {test_metrics[f'{dataset}:top1']}")
    print(f"[{dataset}] Test retain accuracy: {test_metrics[f'{control_dataset}:top1']}")

    negation_accuracies[dataset] = {
        "test": test_metrics[f"{dataset}:top1"],
        "test_control": test_metrics[f"{control_dataset}:top1"],
        "val": val_metrics,
    }

# Create dynamic suffix based on experiment parameters
if args.layer_wise_zeroing:
    suffix = f"_layer_zeroed_thresh_{args.shallow_threshold}.json"
else:
    suffix = ".json"

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/merge_result/negations_negmerge{suffix}"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/merge_result/linear_negations_negmerge{suffix}"
else:
    raise ValueError("finetuning-mode must be either 'standard' or 'linear' for negmerge evaluation")

os.makedirs(os.path.dirname(save_file), exist_ok=True)

with open(save_file, "w") as f:
    json.dump(negation_accuracies, f, indent=4)

print(f"Results successfully saved to: {save_file}")
