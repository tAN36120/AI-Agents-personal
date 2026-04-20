# Experiment 4 — Layer-wise consensus thresholds

This experiment evaluates the **layer-wise zeroing** hypothesis from the paper: zero NegMerge task-vector updates in shallow layers and apply NegMerge only to deeper layers.

## HPC dataset setup (recommended for reproducibility)

For ImageNet2012 and Stanford Cars, use centrally managed HPC datasets so your runs use the same dataset version as the paper.

```bash
module load common-dataset-loader
common-dataset-loader avail
common-dataset-loader load <dataset_path>
ln -s /home/<username>/hpc-shared-data/<dataset_path> /scratch/<username>/
```

For this experiment, load both dataset paths (ImageNet2012 + Stanford Cars), then set:

- `--data-location /scratch/<username>`

## 0) Map HuggingFace artifacts to `--results-db` layout

NegMerge expects this structure:

```text
<results_db>/<finetuning_mode>/<model>/
  zeroshot.pt (or linear_zeroshot.pt)
  zeroshot_accuracies.json
  *checkpoints*/<Dataset>Val/finetuned.pt
```

Use the mapping script:

```bash
python tools/prepare_results_db_from_hf.py \
  --source-root /path/to/hf_downloaded/clip-finetuned-exp1 \
  --results-db /path/to/Experiment_1/results \
  --finetuning-mode standard \
  --model ViT-B-32
```

- By default it creates symlinks (fast, no duplication).
- Add `--copy` if symlinks are not allowed on your environment.

## 1) Run a single threshold (custom dataset list)

From `COMP6258-Negmerge-main/Experiment_4`:

```bash
python src/negmerge.py \
  --model ViT-B-32 \
  --finetuning-mode standard \
  --results-db /path/to/Experiment_1/results \
  --data-location /scratch/<username> \
  --openclip-cachedir /path/to/open_clip_cache \
  --layer-wise-zeroing \
  --shallow-threshold 3 \
  --merge-datasets Cars,SVHN
```

This writes outputs to:

- `/path/to/Experiment_1/results/standard/ViT-B-32/merge_result/negations_negmerge_layer_zeroed_thresh_3.json`

## 2) Sweep thresholds (-1 to 11)

```bash
for t in $(seq -1 11); do
  python src/negmerge.py \
    --model ViT-B-32 \
    --finetuning-mode standard \
    --results-db /path/to/Experiment_1/results \
    --data-location /scratch/<username> \
    --openclip-cachedir /path/to/open_clip_cache \
    --layer-wise-zeroing \
    --shallow-threshold "$t" \
    --merge-datasets Cars,SVHN
done
```

You can change datasets directly, e.g. `--merge-datasets Cars` or `--merge-datasets SVHN`.

## 3) Generate result comparison table (no matplotlib required)

```bash
python tools/compare_exp4_results.py \
  --results-dir ./results \
  --datasets Cars,SVHN \
  --retain-ratio 0.95
```

Outputs:

- `./results/exp4_comparison_report.json`
- `./results/exp4_comparison_report.csv`

## 4) Optional plotting (requires matplotlib)

```bash
python results_analysis.py
```

Default `RESULTS_DIR` is `./results`.
