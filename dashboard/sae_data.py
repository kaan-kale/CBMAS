"""Data loading helpers for the SAE feature inspection dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_ANALYSIS_FILES = (
    "comparison_summary.csv",
    "recurring_top_features.csv",
    "global_recurring_features.csv",
)


def validate_run_dir(run_dir: Path) -> Path:
    """Return a resolved artifact run directory or raise a useful error."""
    resolved = run_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {resolved}")

    missing = [name for name in REQUIRED_ANALYSIS_FILES if not (resolved / "analysis" / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing analysis files in {resolved / 'analysis'}: {joined}")
    return resolved


def load_analysis_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the generated analysis CSVs for one artifact run."""
    resolved = validate_run_dir(run_dir)
    analysis_dir = resolved / "analysis"
    return {
        "comparison": pd.read_csv(analysis_dir / "comparison_summary.csv"),
        "recurring": pd.read_csv(analysis_dir / "recurring_top_features.csv"),
        "global": pd.read_csv(analysis_dir / "global_recurring_features.csv"),
    }


def load_prompt_map(run_dir: Path) -> dict[int, str]:
    """Collect prompt IDs and text from metadata files in a run."""
    resolved = run_dir.expanduser().resolve()
    prompts: dict[int, str] = {}
    for metadata_path in sorted(resolved.glob("source_*_read_*/alpha_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for prompt in metadata.get("prompts", []):
            prompt_id = int(prompt["prompt_id"])
            text = str(prompt.get("prompt", ""))
            if text:
                prompts.setdefault(prompt_id, text)
    return prompts


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    """Load representative metadata shared by the experiment groups."""
    resolved = run_dir.expanduser().resolve()
    metadata_paths = sorted(resolved.glob("source_*_read_*/alpha_*/metadata.json"))
    if not metadata_paths:
        return {}
    return json.loads(metadata_paths[0].read_text(encoding="utf-8"))


def load_prompt_feature_details(run_dir: Path) -> pd.DataFrame:
    """Flatten per-prompt top changed feature summaries across all groups."""
    resolved = run_dir.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(resolved.glob("source_*_read_*/alpha_*/summaries/sae_feature_summary.json")):
        group_dir = summary_path.parents[1]
        metadata_path = group_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        prompt_texts = {
            int(prompt["prompt_id"]): str(prompt.get("prompt", ""))
            for prompt in metadata.get("prompts", [])
        }
        for prompt_summary in summaries:
            prompt_id = int(prompt_summary.get("prompt_id", -1))
            for rank, feature in enumerate(prompt_summary.get("top_changed_features", []), start=1):
                mean_delta = float(feature["mean_delta"])
                rows.append(
                    {
                        "source_layer": int(metadata.get("source_layer", -1)),
                        "read_layer": int(metadata.get("read_layer", -1)),
                        "alpha": float(metadata.get("alpha", 0.0)),
                        "prompt_id": prompt_id,
                        "prompt": prompt_texts.get(prompt_id, ""),
                        "feature_id": int(feature["feature"]),
                        "rank": rank,
                        "abs_mean_delta": float(feature["abs_mean_delta"]),
                        "mean_delta": mean_delta,
                        "direction": "increase" if mean_delta > 0 else "decrease" if mean_delta < 0 else "flat",
                    }
                )
    return pd.DataFrame(rows)


def neuronpedia_feature_url(read_layer: int, feature_id: int) -> str:
    """Build a Gemma Scope residual-stream feature URL for Gemma 2 2B."""
    return (
        "https://www.neuronpedia.org/gemma-2-2b/"
        f"{int(read_layer)}-gemmascope-res-16k/{int(feature_id)}"
    )
