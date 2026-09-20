"""Command-line entry point for the publication citation benchmark."""

import argparse
from pathlib import Path

from src.models_config import (
    get_model_doc_text,
    get_model_normalize,
    get_model_q_text,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("test-ranking", "test-bm25"))
    parser.add_argument("--test-dataset-path", required=True)
    parser.add_argument("--test-data-type", required=True)
    parser.add_argument("--test-model-path")
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    args = parser.parse_args()

    if args.mode == "test-bm25":
        from src.models.test_bm25 import test as test_bm25

        test_bm25(args.test_dataset_path, args.test_data_type)
        return

    if not args.test_model_path:
        parser.error("--test-model-path is required for test-ranking")

    from src.models.test_model import test_model_with_ranking

    # Kept from cross-corpus_testing: model-specific prefixes and normalization.
    model_name = Path(args.test_model_path).name
    test_model_with_ranking(
        args.test_model_path,
        args.test_dataset_path,
        args.test_data_type,
        args.pooling,
        q_text=get_model_q_text(model_name),
        doc_text=get_model_doc_text(model_name),
        normalize=get_model_normalize(model_name),
    )


if __name__ == "__main__":
    main()
