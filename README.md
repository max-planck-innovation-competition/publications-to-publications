# Models Benchmarking on Citation Data

This repository benchmarks our models against several baseline models on publication-to-publication citation data from [SciRepEval](https://huggingface.co/datasets/allenai/scirepeval).

## Overview

The **baseline models** are mostly BERT-based and downloaded from **Hugging Face**:

- BERT-large (uncased)
- BERT for Patents
- SciBERT (uncased)
- SPECTER
- SPECTER 2.0
- gte-large
- PatentSBERTa
- bge-large-en-v1.5
- e5-large-v2
- patembed-large

Additionally, we include **BM25** as a classical baseline.

Our models are:

- PatSPECTER
- PaECTER
- PubPaECTER (disabled by default; see [Model Configuration](#model-configuration))

## Model Configuration

All model information is centrally managed in a single file: **`models_config.yaml`** (in the project root). Each model entry contains:

- `huggingface_id` – Hugging Face repo ID for downloading
- `display_name` – Name shown in tables
- `directory_name` – Folder name under `pretrained_models/` and used for result files

Some models also have query and document text prefixes or an embedding normalization setting. The Python helper `src/models_config.py` reads this file.

### Adding a new model

1. Add the model entry to `models_config.yaml`.
2. Run `make generate-makefile-models` to update `models.mk`.

PubPaECTER has no public Hugging Face download, so it is disabled by default. To include it, place the model files under `pretrained_models/pubpaecter/`, set `enabled: true` in `models_config.yaml`, and regenerate `models.mk`.

## Evaluation Data

We evaluate our models and baselines on three SciRepEval citation tasks using rank-aware metrics.

| Dataset | Source split |
| --- | --- |
| `cite_prediction` | `validation` |
| `cite_prediction_new` | `validation` |
| `cite_prediction_aug2023refresh` | `train` |

The data is read directly from Hugging Face. The `cite_prediction_aug2023refresh` task has no validation split, so its result is **not** a validation result.

**The rank-aware metrics used:**

- RFR (Rank First Relevant)
- MAP (Mean Average Precision)
- MRR@10 (Mean Reciprocal Rank at 10)
- NDCG@10 (Normalized Discounted Cumulative Gain at 10)

## Prerequisites

- Python 3.12, Make, Git, and Git LFS
- Slurm access if you want to submit GPU jobs

From the project root, install the Python packages:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
git lfs version
```

The Makefile uses `python3` from the active environment. If needed, set `PYTHON=/absolute/path/to/python` when running Make.

## Benchmarking

To benchmark our models against baselines using rank-aware metrics, we need to:

- get the test data
- download models
- test them on the three datasets

### Step 1: Get the test data

Run:

```sh
make prepare-data
```

This creates one file for each task under `data/science_test/<task>/split_test.jsonl`. Each line contains one query publication and its positive and negative candidates.

By default, the script selects 1,000 unique queries per task with seed 42. To use a different count or seed, run `make prepare-data QUERY_COUNT=100 SEED=42`, for example.

### Step 2: Download all models

Run:

```sh
make download-pretrained-models
```

This downloads the enabled models to `pretrained_models/`. Git LFS is needed for the model weights. The downloaded files are not stored in this repository.

### Step 3: Run tests

To run all datasets locally, one after another:

```sh
make test-science-citations
```

To submit the model runs as Slurm jobs instead:

```sh
make submit-science-citations PYTHON=/absolute/path/to/python
squeue --me
```

With 14 enabled models and three datasets, the Slurm command submits 42 jobs. Each job runs both CLS and mean pooling. BM25 runs on the node where you call Make. Check that all jobs have finished before generating tables.

## Results

The benchmarking results are stored under `output/`:

- `output/science_cite_prediction/ranking/`
- `output/science_cite_prediction_new/ranking/`
- `output/science_cite_prediction_aug2023refresh/ranking/`

Since we evaluate performance using two pooling methods — `cls` and `mean` — each ranking folder has a subfolder for each method. BM25 results are saved in a separate `bm25/` subfolder.

To generate CSV and LaTeX tables from these results, run:

```sh
make tables
```

The tables are saved in:

- `output/science_cite_prediction/table/`
- `output/science_cite_prediction_new/table/`
- `output/science_cite_prediction_aug2023refresh/table/`
- `output/science_cite_average/table/`

The average table takes the mean of the three dataset results for each model and pooling method. BM25 is not included in the model tables.
