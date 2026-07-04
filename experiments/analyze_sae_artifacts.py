"""Analyze SAE artifact runs produced by experiments.steering_artifacts.

This script is intentionally lightweight: it reads the saved JSON summaries from
a downloaded Modal volume run and produces comparison tables/plots locally.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cbmas_matplotlib"))

import matplotlib.pyplot as plt


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _try_read_json(path: Path) -> Any | None:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return _read_json(path)
    except json.JSONDecodeError:
        return None


def _alpha_from_label(label: str) -> float | None:
    if not label.startswith("alpha_"):
        return None
    raw = label.removeprefix("alpha_").replace("neg", "-").replace("p", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def _infer_metadata_from_path(summary_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for part in summary_path.parts:
        layer_match = re.fullmatch(r"source_(\d+)_read_(\d+)", part)
        if layer_match:
            metadata["source_layer"] = int(layer_match.group(1))
            metadata["read_layer"] = int(layer_match.group(2))
        alpha = _alpha_from_label(part)
        if alpha is not None:
            metadata["alpha"] = alpha
    return metadata


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _discover_summary_files(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.rglob("summaries/sae_feature_summary.json") if path.is_file())


def _load_artifact_group(summary_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    group_dir = summary_path.parents[1]
    metadata_path = group_dir / "metadata.json"
    metadata = _try_read_json(metadata_path) or {}
    inferred_metadata = _infer_metadata_from_path(summary_path)
    metadata = {**inferred_metadata, **metadata}
    feature_summaries = _read_json(summary_path)
    if not isinstance(feature_summaries, list):
        raise ValueError(f"Expected a list in {summary_path}")
    return metadata, feature_summaries


def _summarize_group(metadata: dict[str, Any], feature_summaries: list[dict[str, Any]], summary_path: Path) -> dict[str, Any]:
    read_layer = int(metadata.get("read_layer", -1))
    alpha = float(metadata.get("alpha", 0.0))
    source_layer = int(metadata.get("source_layer", -1))

    unsteered_l0 = [float(row["unsteered_l0_mean"]) for row in feature_summaries]
    steered_l0 = [float(row["steered_l0_mean"]) for row in feature_summaries]
    unsteered_norm = [float(row["unsteered_norm_mean"]) for row in feature_summaries]
    steered_norm = [float(row["steered_norm_mean"]) for row in feature_summaries]
    delta_norm = [float(row["delta_norm_mean"]) for row in feature_summaries]
    feature_dim = int(feature_summaries[0].get("feature_dim", 0)) if feature_summaries else 0

    mean_unsteered_l0 = _safe_mean(unsteered_l0)
    mean_steered_l0 = _safe_mean(steered_l0)
    mean_unsteered_norm = _safe_mean(unsteered_norm)
    mean_steered_norm = _safe_mean(steered_norm)

    return {
        "source_layer": source_layer,
        "read_layer": read_layer,
        "alpha": alpha,
        "num_prompts": len(feature_summaries),
        "feature_dim": feature_dim,
        "mean_sae_delta_norm": _safe_mean(delta_norm),
        "max_sae_delta_norm": max(delta_norm) if delta_norm else 0.0,
        "mean_unsteered_l0": mean_unsteered_l0,
        "mean_steered_l0": mean_steered_l0,
        "mean_l0_change": mean_steered_l0 - mean_unsteered_l0,
        "mean_unsteered_norm": mean_unsteered_norm,
        "mean_steered_norm": mean_steered_norm,
        "mean_norm_change": mean_steered_norm - mean_unsteered_norm,
        "summary_path": str(summary_path),
    }


def _feature_rows(
    metadata: dict[str, Any],
    feature_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    read_layer = int(metadata.get("read_layer", -1))
    alpha = float(metadata.get("alpha", 0.0))
    source_layer = int(metadata.get("source_layer", -1))

    by_feature: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prompt_summary in feature_summaries:
        prompt_id = int(prompt_summary.get("prompt_id", -1))
        for feature in prompt_summary.get("top_changed_features", []):
            by_feature[int(feature["feature"])].append(
                {
                    "prompt_id": prompt_id,
                    "abs_mean_delta": float(feature["abs_mean_delta"]),
                    "mean_delta": float(feature["mean_delta"]),
                }
            )

    rows = []
    for feature_id, hits in by_feature.items():
        prompt_ids = sorted({hit["prompt_id"] for hit in hits})
        mean_delta = _safe_mean([hit["mean_delta"] for hit in hits])
        positive_count = sum(1 for hit in hits if hit["mean_delta"] > 0)
        negative_count = sum(1 for hit in hits if hit["mean_delta"] < 0)
        if positive_count and negative_count:
            direction = "mixed"
        elif positive_count:
            direction = "increase"
        elif negative_count:
            direction = "decrease"
        else:
            direction = "flat"

        rows.append(
            {
                "source_layer": source_layer,
                "read_layer": read_layer,
                "alpha": alpha,
                "feature_id": feature_id,
                "occurrence_count": len(hits),
                "prompt_count": len(prompt_ids),
                "prompt_ids": ",".join(str(pid) for pid in prompt_ids),
                "mean_abs_delta": _safe_mean([hit["abs_mean_delta"] for hit in hits]),
                "mean_delta": mean_delta,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "direction": direction,
            }
        )
    return sorted(rows, key=lambda row: (-row["prompt_count"], -row["mean_abs_delta"], row["feature_id"]))


def _global_feature_rows(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        grouped[(int(row["read_layer"]), int(row["feature_id"]))].append(row)

    rows = []
    for (read_layer, feature_id), hits in grouped.items():
        alpha_values = sorted({float(hit["alpha"]) for hit in hits})
        prompt_count_total = sum(int(hit["prompt_count"]) for hit in hits)
        mean_delta = _safe_mean([float(hit["mean_delta"]) for hit in hits])
        positive_count = sum(int(hit["positive_count"]) for hit in hits)
        negative_count = sum(int(hit["negative_count"]) for hit in hits)
        if positive_count and negative_count:
            direction = "mixed"
        elif positive_count:
            direction = "increase"
        elif negative_count:
            direction = "decrease"
        else:
            direction = "flat"

        rows.append(
            {
                "read_layer": read_layer,
                "feature_id": feature_id,
                "alpha_count": len(alpha_values),
                "alpha_values": ",".join(f"{alpha:g}" for alpha in alpha_values),
                "prompt_count_total": prompt_count_total,
                "mean_abs_delta": _safe_mean([float(hit["mean_abs_delta"]) for hit in hits]),
                "mean_delta": mean_delta,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "direction": direction,
            }
        )
    return sorted(rows, key=lambda row: (-row["alpha_count"], -row["prompt_count_total"], -row["mean_abs_delta"]))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(summary_rows: list[dict[str, Any]], metric: str, ylabel: str, title: str, out_path: Path) -> None:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_layer[int(row["read_layer"])].append(row)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for read_layer, rows in sorted(by_layer.items()):
        sorted_rows = sorted(rows, key=lambda row: float(row["alpha"]))
        ax.plot(
            [float(row["alpha"]) for row in sorted_rows],
            [float(row[metric]) for row in sorted_rows],
            marker="o",
            linewidth=2,
            label=f"read layer {read_layer}",
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(r"Steering coefficient $\alpha$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_run(run_dir: Path, out_dir: Path | None = None) -> dict[str, str]:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    output_dir = (out_dir.expanduser().resolve() if out_dir else run_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_files = _discover_summary_files(run_dir)
    if not summary_files:
        raise ValueError(f"No sae_feature_summary.json files found under {run_dir}")

    summary_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for summary_path in summary_files:
        metadata, feature_summaries = _load_artifact_group(summary_path)
        if not feature_summaries:
            continue
        summary_rows.append(_summarize_group(metadata, feature_summaries, summary_path))
        feature_rows.extend(_feature_rows(metadata, feature_summaries))

    if not summary_rows:
        raise ValueError(f"SAE summary files were found, but none contained feature summaries under {run_dir}")

    global_feature_rows = _global_feature_rows(feature_rows)

    comparison_summary_path = output_dir / "comparison_summary.csv"
    recurring_features_path = output_dir / "recurring_top_features.csv"
    global_features_path = output_dir / "global_recurring_features.csv"

    _write_csv(
        comparison_summary_path,
        sorted(summary_rows, key=lambda row: (int(row["read_layer"]), float(row["alpha"]))),
        [
            "source_layer",
            "read_layer",
            "alpha",
            "num_prompts",
            "feature_dim",
            "mean_sae_delta_norm",
            "max_sae_delta_norm",
            "mean_unsteered_l0",
            "mean_steered_l0",
            "mean_l0_change",
            "mean_unsteered_norm",
            "mean_steered_norm",
            "mean_norm_change",
            "summary_path",
        ],
    )
    _write_csv(
        recurring_features_path,
        feature_rows,
        [
            "source_layer",
            "read_layer",
            "alpha",
            "feature_id",
            "occurrence_count",
            "prompt_count",
            "prompt_ids",
            "mean_abs_delta",
            "mean_delta",
            "positive_count",
            "negative_count",
            "direction",
        ],
    )
    _write_csv(
        global_features_path,
        global_feature_rows,
        [
            "read_layer",
            "feature_id",
            "alpha_count",
            "alpha_values",
            "prompt_count_total",
            "mean_abs_delta",
            "mean_delta",
            "positive_count",
            "negative_count",
            "direction",
        ],
    )

    delta_plot = output_dir / "compare_alpha_vs_sae_delta_norm.png"
    l0_plot = output_dir / "compare_alpha_vs_l0_change.png"
    norm_plot = output_dir / "compare_alpha_vs_norm_change.png"
    _plot_metric(
        summary_rows,
        "mean_sae_delta_norm",
        "Mean SAE delta norm",
        "Alpha vs Mean SAE Feature Delta Norm",
        delta_plot,
    )
    _plot_metric(
        summary_rows,
        "mean_l0_change",
        "Mean L0 change",
        "Alpha vs SAE L0 Change",
        l0_plot,
    )
    _plot_metric(
        summary_rows,
        "mean_norm_change",
        "Mean feature norm change",
        "Alpha vs SAE Feature Norm Change",
        norm_plot,
    )

    report_path = output_dir / "analysis_summary.md"
    best_delta = max(summary_rows, key=lambda row: float(row["mean_sae_delta_norm"]))
    top_global = global_feature_rows[:10]
    report = [
        "# SAE Artifact Analysis",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Groups analyzed: `{len(summary_rows)}`",
        f"- Best mean SAE delta norm: read layer `{best_delta['read_layer']}`, alpha `{best_delta['alpha']}`, value `{best_delta['mean_sae_delta_norm']:.4f}`",
        "",
        "## Top Recurring Features",
    ]
    if top_global:
        for row in top_global:
            report.append(
                f"- read layer `{row['read_layer']}`, feature `{row['feature_id']}`: "
                f"alpha_count `{row['alpha_count']}`, prompt_count_total `{row['prompt_count_total']}`, "
                f"direction `{row['direction']}`"
            )
    else:
        report.append("- No recurring features found.")
    report.extend(
        [
            "",
            "## Files",
            f"- `{comparison_summary_path}`",
            f"- `{recurring_features_path}`",
            f"- `{global_features_path}`",
            f"- `{delta_plot}`",
            f"- `{l0_plot}`",
            f"- `{norm_plot}`",
        ]
    )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    outputs = {
        "analysis_dir": str(output_dir),
        "comparison_summary": str(comparison_summary_path),
        "recurring_top_features": str(recurring_features_path),
        "global_recurring_features": str(global_features_path),
        "delta_norm_plot": str(delta_plot),
        "l0_change_plot": str(l0_plot),
        "norm_change_plot": str(norm_plot),
        "report": str(report_path),
    }
    print(json.dumps(outputs, indent=2))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze SAE feature summaries from a steering artifact run.")
    parser.add_argument("--run-dir", required=True, help="Downloaded steering run directory.")
    parser.add_argument("--out-dir", default="", help="Optional output directory. Defaults to <run-dir>/analysis.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    analyze_run(Path(args.run_dir), Path(args.out_dir) if args.out_dir else None)


if __name__ == "__main__":
    main()
