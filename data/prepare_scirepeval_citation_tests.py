"""Prepare a citation-ranking test set from a SciRepEval split."""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "allenai/scirepeval"
OUTPUT_FILES = {
    "cite_prediction": "data/science_test/cite_prediction/split_test.jsonl",
    "cite_prediction_new": (
        "data/science_test/cite_prediction_new/split_test.jsonl"
    ),
    "cite_prediction_aug2023refresh": (
        "data/science_test/cite_prediction_aug2023refresh/split_test.jsonl"
    ),
}


def get_document_id(document):
    """Get the corpus or document ID used to deduplicate papers."""
    if document.get("corpus_id") is not None:
        return "corpus:" + str(document["corpus_id"])
    if document.get("doc_id") is not None:
        return "doc:" + str(document["doc_id"])
    raise ValueError("Document has neither corpus_id nor doc_id")


def has_text(document):
    """Check that a paper has a title or abstract."""
    if not document:
        return False
    title = document.get("title") or ""
    abstract = document.get("abstract") or ""
    return bool(title.strip() or abstract.strip())


def clean_document(document):
    """Keep required text fields and available source identifiers."""
    cleaned = {
        "title": document.get("title"),
        "abstract": document.get("abstract"),
    }
    for field in ("corpus_id", "doc_id", "sha", "score"):
        if document.get(field) is not None:
            cleaned[field] = document[field]
    return cleaned


def find_eligible_queries(dataset):
    """Find query IDs that have both positive and negative candidates."""
    candidates = {}

    for row in dataset:
        query = row["query"]
        if not has_text(query):
            continue

        query_id = get_document_id(query)
        group = candidates.setdefault(query_id, {"pos": set(), "neg": set()})

        if has_text(row["pos"]):
            group["pos"].add(get_document_id(row["pos"]))
        if has_text(row["neg"]):
            group["neg"].add(get_document_id(row["neg"]))

    eligible = []
    for query_id, group in candidates.items():
        positive_ids = group["pos"] - {query_id}
        negative_ids = group["neg"] - positive_ids - {query_id}
        if positive_ids and negative_ids:
            eligible.append(query_id)

    return sorted(eligible)


def select_queries(eligible_queries, query_count, seed):
    """Select the requested number of query IDs reproducibly."""
    if len(eligible_queries) < query_count:
        raise ValueError(
            f"Only {len(eligible_queries)} eligible queries are available; "
            f"cannot select {query_count}."
        )
    return set(random.Random(seed).sample(eligible_queries, query_count))


def deduplicate_queries(dataset, selected_ids):
    """Collect each selected query and its unique candidate papers."""
    groups = {
        query_id: {"query": None, "pos": {}, "neg": {}}
        for query_id in selected_ids
    }

    for row in dataset:
        query = row["query"]
        if not has_text(query):
            continue

        query_id = get_document_id(query)
        if query_id not in groups:
            continue

        group = groups[query_id]
        if group["query"] is None:
            group["query"] = clean_document(query)

        for label in ("pos", "neg"):
            document = row[label]
            if not has_text(document):
                continue
            document_id = get_document_id(document)
            if document_id != query_id:
                group[label][document_id] = clean_document(document)

    prepared = []
    for query_id in sorted(groups):
        group = groups[query_id]

        # If a paper has both labels, retain it only as a positive candidate.
        for document_id in group["pos"]:
            group["neg"].pop(document_id, None)

        prepared.append({
            "query": group["query"],
            "pos": [group["pos"][key] for key in sorted(group["pos"])],
            "neg": [group["neg"][key] for key in sorted(group["neg"])],
        })

    return prepared


def write_jsonl(records, output_path):
    """Write one query group per JSONL line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_dataset(config, split, query_count, seed):
    """Stream one split, select queries, and write the test JSONL file."""
    project_root = Path(__file__).resolve().parents[1]
    dataset = load_dataset(
        DATASET_ID,
        config,
        split=split,
        streaming=True,
        cache_dir=str(project_root / "data_cache" / "scirepeval"),
    )

    eligible_queries = find_eligible_queries(dataset)
    selected_ids = select_queries(eligible_queries, query_count, seed)
    records = deduplicate_queries(dataset, selected_ids)
    output_path = project_root / OUTPUT_FILES[config]
    write_jsonl(records, output_path)
    print(f"Wrote {len(records)} queries to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=OUTPUT_FILES, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_dataset(args.config, args.split, args.query_count, args.seed)


if __name__ == "__main__":
    main()
