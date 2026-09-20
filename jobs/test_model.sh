#!/usr/bin/env bash
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=92500
#SBATCH --time=1:00:00

set -euo pipefail

MODEL_NAME=$1
TEST_DATA_TYPE=$2
TEST_DATA_PATH=$3
PYTHON=$4
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

cd "$PROJECT_ROOT"

# Same two pooling runs as the original cross-corpus Slurm job.
for pooling in cls mean; do
    srun "$PYTHON" -m src.main test-ranking \
        --test-model-path "$PROJECT_ROOT/pretrained_models/$MODEL_NAME" \
        --test-data-type "$TEST_DATA_TYPE" \
        --test-dataset-path "$TEST_DATA_PATH" \
        --pooling "$pooling"
done
