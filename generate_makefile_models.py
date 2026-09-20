#!/usr/bin/env python3
"""
Generate ``models.mk`` from ``models_config.yaml``.

``models.mk`` defines the enabled ``MODELS`` and their public Hugging Face IDs.
The Makefile uses those IDs for one-model or all-model downloads. Regenerate
this file whenever ``models_config.yaml`` changes.

Usage:
    python3 generate_makefile_models.py
"""
from pathlib import Path

from src.models_config import get_model_directory_names, get_model_huggingface_ids

OUTPUT_FILE = Path(__file__).parent / "models.mk"


def build_models_mk() -> str:
    """Build the contents of models.mk from the central YAML config."""
    directory_names = get_model_directory_names()
    huggingface_ids = get_model_huggingface_ids()

    lines = [
        "# ============================================================",
        "# AUTO-GENERATED from models_config.yaml — DO NOT EDIT BY HAND.",
        "# Regenerate with: python3 generate_makefile_models.py",
        "# ============================================================",
        "",
        "# ============",
        "# MODEL LIST",
        "# ============",
        "MODELS = \\",
    ]
    for i, dir_name in enumerate(directory_names):
        suffix = "" if i == len(directory_names) - 1 else " \\"
        lines.append(f"\t{dir_name}{suffix}")

    lines += ["", "# Public Hugging Face IDs used by make download-model."]
    for dir_name, hf_id in zip(directory_names, huggingface_ids.values()):
        # Added for the standalone repo: a single model can be downloaded by name.
        if hf_id:
            lines.append(f"HF_ID_{dir_name} := {hf_id}")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUTPUT_FILE.write_text(build_models_mk())
    print(f"Wrote {OUTPUT_FILE}")
