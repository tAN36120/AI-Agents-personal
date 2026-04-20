#!/bin/bash
#SBATCH --job-name=negmerge_exp4
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-13
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err

# ID=1 -> threshold=-1, ... , ID=13 -> threshold=11
THRESHOLD=$((SLURM_ARRAY_TASK_ID - 2))
echo "Array task ${SLURM_ARRAY_TASK_ID}: Running Experiment 4 with shallow-threshold=${THRESHOLD}"

source ~/.bashrc
conda activate negmerge-clip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$PYTHONPATH:$PWD"
mkdir -p logs

# Update these paths for your environment
EXP1_RESULTS_DB="/path/to/Experiment_1/results"
OPENCLIP_CACHE="${HOME}/openclip-cachedir/open_clip"

# Load centrally managed datasets from HPC shared storage (same dataset version as paper)
module load common-dataset-loader
common-dataset-loader avail

# Set dataset identifiers provided by your HPC admin/docs
IMAGENET_DATASET_PATH="imagenet2012"
CARS_DATASET_PATH="stanford_cars"

common-dataset-loader load "${IMAGENET_DATASET_PATH}"
common-dataset-loader load "${CARS_DATASET_PATH}"

mkdir -p "/scratch/${USER}"
ln -sfn "/home/${USER}/hpc-shared-data/${IMAGENET_DATASET_PATH}" "/scratch/${USER}/${IMAGENET_DATASET_PATH}"
ln -sfn "/home/${USER}/hpc-shared-data/${CARS_DATASET_PATH}" "/scratch/${USER}/${CARS_DATASET_PATH}"

DATA_ROOT="/scratch/${USER}"

python src/negmerge.py \
    --model ViT-B-32 \
    --finetuning-mode standard \
    --layer-wise-zeroing \
    --shallow-threshold "${THRESHOLD}" \
    --merge-datasets Cars,SVHN \
    --results-db "${EXP1_RESULTS_DB}" \
    --data-location "${DATA_ROOT}" \
    --openclip-cachedir "${OPENCLIP_CACHE}"
