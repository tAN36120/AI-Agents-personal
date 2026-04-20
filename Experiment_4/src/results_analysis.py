import json
import os
import glob
import re
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it or use tools/compare_exp4_results.py for table-only comparison."
    ) from exc

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
DATASETS = ["Cars", "SVHN"]

# ── Collect all threshold JSON files ─────────────────────────────────────────
pattern = os.path.join(RESULTS_DIR, "negations_negmerge_layer_zeroed_thresh_*.json")
files = glob.glob(pattern)

records = {}  # threshold (int) -> {dataset -> {test, test_control}}
baseline_record = {}

for fpath in files:
    match = re.search(r"thresh_([-\d]+)\.json$", fpath)
    if not match:
        continue
    threshold = int(match.group(1))

    with open(fpath) as f:
        data = json.load(f)

    # Treat threshold 0 (or -1) as the baseline
    if threshold <= 0:
        for ds in DATASETS:
            if ds in data:
                baseline_record[ds] = {
                    "forget": data[ds]["test"],
                    "retain": data[ds]["test_control"],
                }
    else:
        records[threshold] = {}
        for ds in DATASETS:
            if ds in data:
                records[threshold][ds] = {
                    "forget": data[ds]["test"],
                    "retain": data[ds]["test_control"],
                }

# Fallback: if baseline wasn't in thresh_0.json, check for pure negmerge.json
if not baseline_record:
    baseline_path = os.path.join(RESULTS_DIR, "negations_negmerge.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            data = json.load(f)
        for ds in DATASETS:
            if ds in data:
                baseline_record[ds] = {
                    "forget": data[ds]["test"],
                    "retain": data[ds]["test_control"],
                }

# ── Sort thresholds numerically ──────────────────────────────────────────────
int_thresholds = sorted([k for k in records.keys() if 1 <= k <= 11])

print("=" * 70)
print(f"{'Threshold':<12} {'Dataset':<8} {'Forget Acc':>12} {'Retain Acc':>12}")
print("=" * 70)

if baseline_record:
    for ds in DATASETS:
        if ds in baseline_record:
            r = baseline_record[ds]
            print(f"{'baseline (0)':<12} {ds:<8} {r['forget']:>12.4f} {r['retain']:>12.4f}")

for thresh in int_thresholds:
    for ds in DATASETS:
        if ds in records[thresh]:
            r = records[thresh][ds]
            print(f"{thresh:<12} {ds:<8} {r['forget']:>12.4f} {r['retain']:>12.4f}")

print("=" * 70)

# ── Plot style ───────────────────────────────────────────────────────────────
colors = {"forget": "#e05c5c", "retain": "#4a90d9"}

def save_single_dataset_plot(ds, output_name):
    fig, ax = plt.subplots(figsize=(7, 5))

    x_vals = int_thresholds
    x_plot = [t for t in x_vals if ds in records[t]]
    forget_vals = [records[t][ds]["forget"] * 100 for t in x_plot]
    retain_vals = [records[t][ds]["retain"] * 100 for t in x_plot]

    # Main curves
    ax.plot(
        x_plot, forget_vals,
        marker="o", color=colors["forget"],
        linewidth=2, markersize=7,
        label=r"Forget Acc $D_f$ ↓"
    )
    ax.plot(
        x_plot, retain_vals,
        marker="s", color=colors["retain"],
        linewidth=2, markersize=7,
        label=r"Retain Acc $D_r$ ↑"
    )

    # Point annotations
    for x, fy, ry in zip(x_plot, forget_vals, retain_vals):
        ax.annotate(
            f"{fy:.1f}", (x, fy),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=8, color=colors["forget"]
        )
        ax.annotate(
            f"{ry:.1f}", (x, ry),
            textcoords="offset points", xytext=(0, -14),
            ha="center", fontsize=8, color=colors["retain"]
        )

    # Baseline lines
    if baseline_record and ds in baseline_record:
        bf = baseline_record[ds]["forget"] * 100
        br = baseline_record[ds]["retain"] * 100

        ax.axhline(
            bf, linestyle="--", color=colors["forget"],
            alpha=0.5, linewidth=1.5,
            label=f"Baseline Forget ({bf:.1f}%)"
        )
        ax.axhline(
            br, linestyle="--", color=colors["retain"],
            alpha=0.5, linewidth=1.5,
            label=f"Baseline Retain ({br:.1f}%)"
        )

    # Labels only, no title
    ax.set_xlabel("Layer cutoff (layers ≤ k zeroed)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xticks(x_plot)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_name}")
    plt.close()

# ── Save two separate figures ────────────────────────────────────────────────
save_single_dataset_plot("Cars", "exp4_layer_zeroing_cars.jpg")
save_single_dataset_plot("SVHN", "exp4_layer_zeroing_svhn.jpg")

# ── Summary: best threshold per dataset ──────────────────────────────────────
print("\n" + "=" * 70)
print("BEST THRESHOLD PER DATASET (lowest forget, subject to retain >= 0.95 * pretrained)")
print("=" * 70)

PRETRAINED_IMAGENET = 0.667
threshold_95 = 0.95 * PRETRAINED_IMAGENET

for ds in DATASETS:
    candidates = [
        (t, records[t][ds])
        for t in int_thresholds
        if ds in records[t] and records[t][ds]["retain"] >= threshold_95
    ]
    if not candidates:
        print(f"  {ds}: No threshold meets the 95% retain constraint ({threshold_95:.4f})")
        continue
    best = min(candidates, key=lambda x: x[1]["forget"])
    print(f"  {ds}: threshold={best[0]}  forget={best[1]['forget']:.4f}  retain={best[1]['retain']:.4f}")