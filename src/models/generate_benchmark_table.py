import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.models_config import (
    get_model_display_names,
    get_model_huggingface_ids,
    get_model_order,
)


model_order = get_model_order()
model_labels = get_model_display_names()
METRICS = (
    ("RFR", "Avg. RFR", False),
    ("MRR@10", "MRR@10", True),
    ("NDCG@10", "nDCG@10", True),
    ("MAP", "MAP", True),
)
SCIENCE_CITE_DATA_TYPES = (
    "science_cite_prediction",
    "science_cite_prediction_new",
    "science_cite_prediction_aug2023refresh",
)


def load_pooling_results(results_dir, pooling):
    """Load one complete summary row for every configured model."""
    result_paths = {}
    for path in results_dir.glob("RerankingEvaluator_*_results.csv"):
        if path.name.endswith("_whole_sample_results.csv"):
            continue
        model_name = path.name.removeprefix(
            "RerankingEvaluator_"
        ).removesuffix("_results.csv")
        result_paths[model_name] = path

    missing_models = [
        model_name
        for model_name in model_order
        if model_name not in result_paths
    ]
    if missing_models:
        raise FileNotFoundError(
            f"Cannot generate the {pooling} table; missing results for: "
            + ", ".join(missing_models)
        )

    required_columns = {metric for metric, _, _ in METRICS}
    result_frames = []

    for model_name in model_order:
        path = result_paths[model_name]
        result = pd.read_csv(path)
        missing_columns = required_columns.difference(result.columns)

        if missing_columns:
            raise ValueError(
                f"{path} is missing required metrics: "
                + ", ".join(sorted(missing_columns))
                + ". Rerun this model with the current evaluator."
            )
        if len(result) != 1:
            raise ValueError(
                f"{path} must contain exactly one aggregate result row"
            )

        result = result.copy()
        result["model"] = model_name
        result_frames.append(result)

    return pd.concat(result_frames, ignore_index=True)


def average_science_citation_results(pooling):
    """Average one pooling method across the three citation datasets."""
    metric_columns = [metric for metric, _, _ in METRICS]
    dataset_results = []

    for data_type in SCIENCE_CITE_DATA_TYPES:
        results_dir = (
            Path(config.Config.OUTPUT_PATH)
            / data_type
            / "ranking"
            / pooling
        )
        results = load_pooling_results(results_dir, pooling)
        dataset_results.append(results[["model"] + metric_columns])

    combined = pd.concat(dataset_results, ignore_index=True)
    return combined.groupby(
        "model",
        as_index=False,
    )[metric_columns].mean()


def scale_percentage_metrics(results):
    """Display MAP, MRR, and NDCG as percentages."""
    results = results.copy()
    for metric in ("MAP", "MRR@10", "NDCG@10"):
        results[metric] = (results[metric] * 100).round(2)
    return results


def generate_csv_table(results_df, output_dir, test_data_type):
    huggingface_ids = get_model_huggingface_ids()
    hf_df = pd.DataFrame.from_dict(
        huggingface_ids,
        orient="index",
    ).reset_index()
    hf_df.columns = ["model", "huggingface_id"]

    results_df = results_df.merge(hf_df, on="model", how="left")
    results_df["url"] = results_df["huggingface_id"].apply(
        lambda value: (
            f"https://huggingface.co/{value}"
            if pd.notna(value)
            else ""
        )
    )
    results_df = results_df.sort_values("RFR_cls", ascending=True)

    csv_columns = ["model_display", "url"]
    for metric, _, _ in METRICS:
        csv_columns.extend([
            f"{metric}_mean",
            f"{metric}_cls",
        ])

    results_df = results_df[csv_columns]
    results_df.to_csv(
        output_dir / f"benchmark_results_table_{test_data_type}.csv",
        index=False,
    )


def generate_latex_table(results_df):
    best_indices = {}

    for metric, _, higher_is_better in METRICS:
        for pooling in ("cls", "mean"):
            column = f"{metric}_{pooling}"
            best_indices[column] = (
                results_df[column].idxmax()
                if higher_is_better
                else results_df[column].idxmin()
            )

    def fmt(column, value, index):
        formatted = f"{value:.2f}"
        if best_indices[column] == index:
            return rf"\textbf{{{formatted}}}"
        return formatted

    line_end = r" \\" + "\n"
    metric_columns = " cc" * len(METRICS)
    latex = rf"\begin{{tabular}}{{l{metric_columns}}}" + "\n"
    latex += "    " + r"\toprule" + "\n"
    latex += "    & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{label}}}"
        for _, label, _ in METRICS
    ) + line_end
    latex += "    " + " ".join(
        rf"\cmidrule(lr){{{2 + 2 * index}-{3 + 2 * index}}}"
        for index in range(len(METRICS))
    ) + "\n"
    latex += "Model & " + " & ".join(
        value
        for _ in METRICS
        for value in ("CLS", "Mean")
    ) + line_end
    latex += "    " + r"\midrule" + "\n"

    for index, row in results_df.iterrows():
        values = []
        for metric, _, _ in METRICS:
            for pooling in ("cls", "mean"):
                column = f"{metric}_{pooling}"
                values.append(fmt(column, row[column], index))

        model_display = row["model_display"]
        latex += (
            f"    {model_display} & "
            + " & ".join(values)
            + line_end
        )

    latex += "    " + r"\bottomrule" + "\n"
    latex += r"\end{tabular}" + "\n"
    return latex


def main():
    parser = argparse.ArgumentParser()
    # Standalone repo: reject patent and cross-corpus output names here.
    parser.add_argument(
        "--test-data-type",
        choices=SCIENCE_CITE_DATA_TYPES + ("science_cite_average",),
        default="science_cite_prediction",
        help="Publication citation dataset, or the three-dataset average",
    )
    args = parser.parse_args()
    test_data_type = args.test_data_type

    if test_data_type == "science_cite_average":
        cls_result_df = average_science_citation_results("cls")
        mean_result_df = average_science_citation_results("mean")
    else:
        input_dir = (
            Path(config.Config.OUTPUT_PATH)
            / test_data_type
            / "ranking"
        )
        cls_result_df = load_pooling_results(input_dir / "cls", "CLS")
        mean_result_df = load_pooling_results(input_dir / "mean", "mean")

    cls_result_df = scale_percentage_metrics(cls_result_df)
    mean_result_df = scale_percentage_metrics(mean_result_df)

    results = cls_result_df.merge(
        mean_result_df,
        on="model",
        suffixes=("_cls", "_mean"),
    )
    results["model"] = pd.Categorical(
        results["model"],
        categories=model_order,
        ordered=True,
    )
    results = results.sort_values("model")
    results["model_display"] = results["model"].map(model_labels)

    output_dir = (
        Path(config.Config.OUTPUT_PATH)
        / test_data_type
        / "table"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_csv_table(results, output_dir, test_data_type)
    latex = generate_latex_table(results)

    output_file = output_dir / "benchmark_results_table.tex"
    with output_file.open("w", encoding="utf-8") as file:
        file.write(latex)

    print(f"Generated LaTeX table saved to: {output_file}")


if __name__ == "__main__":
    main()
