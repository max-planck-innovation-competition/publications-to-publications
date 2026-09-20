BASE_PATH := $(CURDIR)
PYTHON ?= python3

DATA_DIR := $(BASE_PATH)/data
PRETRAINED_DIR := $(BASE_PATH)/pretrained_models
LOG_DIR := $(BASE_PATH)/logs/test_model
QUERY_COUNT ?= 1000
SEED ?= 42
MODEL ?= specter
DATASET ?= science_cite_prediction
TEST_PATH = $(DATA_DIR)/science_test/$(patsubst science_%,%,$(DATASET))/split_test.jsonl

include models.mk

prepare-data: prepare-scirepeval-cite-prediction-data \
	prepare-scirepeval-cite-prediction-new-data \
	prepare-scirepeval-cite-prediction-aug2023refresh-data

prepare-scirepeval-cite-prediction-data:
	$(PYTHON) data/prepare_scirepeval_citation_tests.py \
		--config cite_prediction --split validation \
		--query-count $(QUERY_COUNT) --seed $(SEED)

prepare-scirepeval-cite-prediction-new-data:
	$(PYTHON) data/prepare_scirepeval_citation_tests.py \
		--config cite_prediction_new --split validation \
		--query-count $(QUERY_COUNT) --seed $(SEED)

prepare-scirepeval-cite-prediction-aug2023refresh-data:
	$(PYTHON) data/prepare_scirepeval_citation_tests.py \
		--config cite_prediction_aug2023refresh --split train \
		--query-count $(QUERY_COUNT) --seed $(SEED)

generate-makefile-models:
	$(PYTHON) generate_makefile_models.py

download-model:
	@if [ -f "$(PRETRAINED_DIR)/$(MODEL)/config.json" ]; then \
		echo "Model $(MODEL) already present"; \
	else \
		test -n "$(HF_ID_$(MODEL))" || { echo "Provide model $(MODEL) locally; no public Hugging Face ID"; exit 1; }; \
		git lfs version >/dev/null || { echo "Install Git LFS before downloading models"; exit 1; }; \
		mkdir -p "$(PRETRAINED_DIR)"; \
		git lfs install; \
		git clone --progress "https://huggingface.co/$(HF_ID_$(MODEL))" "$(PRETRAINED_DIR)/$(MODEL)"; \
	fi
	@test -f "$(PRETRAINED_DIR)/$(MODEL)/config.json" || { \
		echo "Incomplete model: $(MODEL) (missing config.json)"; exit 1; }

download-pretrained-models:
	@set -e; for model in $(MODELS); do \
		$(MAKE) download-model MODEL="$$model"; \
	done

check-data:
	@test -f "$(TEST_PATH)" || { \
		echo "Missing $(TEST_PATH); run make prepare-data first"; exit 1; }

check-models:
	@set -e; for model in $(MODELS); do \
		test -f "$(PRETRAINED_DIR)/$$model/config.json" || { \
			echo "Missing model $$model; run make download-pretrained-models first"; exit 1; }; \
	done

test-bm25: check-data
	$(PYTHON) -m src.main test-bm25 \
		--test-data-type "$(DATASET)" --test-dataset-path "$(TEST_PATH)"

test-dataset: check-data check-models
	@$(MAKE) test-bm25 DATASET="$(DATASET)"
	@set -e; for model in $(MODELS); do \
		for pooling in cls mean; do \
			$(PYTHON) -m src.main test-ranking \
				--test-model-path "$(PRETRAINED_DIR)/$$model" \
				--test-data-type "$(DATASET)" \
				--test-dataset-path "$(TEST_PATH)" \
				--pooling "$$pooling"; \
		done; \
	done

test-science-cite-prediction:
	@$(MAKE) test-dataset DATASET=science_cite_prediction

test-science-cite-prediction-new:
	@$(MAKE) test-dataset DATASET=science_cite_prediction_new

test-science-cite-prediction-aug2023refresh:
	@$(MAKE) test-dataset DATASET=science_cite_prediction_aug2023refresh

test-science-citations: test-science-cite-prediction \
	test-science-cite-prediction-new \
	test-science-cite-prediction-aug2023refresh

# Slurm alternative: BM25 runs now, then one GPU job per transformer model.
submit-dataset: check-data check-models
	@command -v sbatch >/dev/null || { echo "sbatch is not available"; exit 1; }
	@$(MAKE) test-bm25 DATASET="$(DATASET)"
	@mkdir -p "$(LOG_DIR)"
	@set -e; for model in $(MODELS); do \
		sbatch --job-name="test-$$model-$(DATASET)" \
			--output="$(LOG_DIR)/test-$$model-$(DATASET)_%j.out" \
			jobs/test_model.sh "$$model" "$(DATASET)" "$(TEST_PATH)" "$(PYTHON)"; \
	done

submit-science-cite-prediction:
	@$(MAKE) submit-dataset DATASET=science_cite_prediction

submit-science-cite-prediction-new:
	@$(MAKE) submit-dataset DATASET=science_cite_prediction_new

submit-science-cite-prediction-aug2023refresh:
	@$(MAKE) submit-dataset DATASET=science_cite_prediction_aug2023refresh

submit-science-citations: submit-science-cite-prediction \
	submit-science-cite-prediction-new \
	submit-science-cite-prediction-aug2023refresh

table:
	$(PYTHON) -m src.models.generate_benchmark_table --test-data-type "$(DATASET)"

# Produces three individual tables and their per-model, per-pooling average.
tables:
	@$(MAKE) table DATASET=science_cite_prediction
	@$(MAKE) table DATASET=science_cite_prediction_new
	@$(MAKE) table DATASET=science_cite_prediction_aug2023refresh
	@$(MAKE) table DATASET=science_cite_average
