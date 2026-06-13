"""Small Gemma alpha sweep for CBMAS-style activation steering diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cbmas_matplotlib"))

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from BRC_Experiment.Modularized.data import load_test_dataset, load_train_dataset
from BRC_Experiment.Modularized.utils import build_hook_name, configure_determinism, get_device
from BRC_Experiment.Modularized.cache import VectorCache
from BRC_Experiment.Modularized.utils import unit_vector
from BRC_Experiment.Modularized.vectors import build_vectors


DEFAULT_ALPHAS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]


@dataclass(frozen=True)
class LayerPair:
    source_layer: int
    read_layer: int


@dataclass
class GemmaSweepConfig:
    model_name: str = "google/gemma-2-2b-it"
    behavior_name: str = "reassurance"
    source_layer: int = 12
    read_layer: int = 20
    source_site: str = "hook_resid_mid"
    read_site: str = "hook_resid_post"
    alpha_values: tuple[float, ...] = tuple(DEFAULT_ALPHAS)
    max_train_prompts: int = 40
    max_eval_prompts: int = 30
    prepend_bos: bool = True
    steer_all_tokens: bool = True
    seed: int = 42
    out_dir: str = "graphs/gemma_alpha_sweep"
    run_id: str = ""
    layer_pairs: tuple[LayerPair, ...] = field(default_factory=tuple)


def _clean_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


def _make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{uuid.uuid4().hex[:4]}"


def _layer_label(source_layer: int, read_layer: int) -> str:
    return f"source_{source_layer}_read_{read_layer}"


def _comparison_label(source_layer: int, read_layer: int) -> str:
    return f"src{source_layer}_read{read_layer}"


def _parse_alpha_values(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_layer_pairs(value: str | None) -> tuple[LayerPair, ...]:
    if not value:
        return tuple()

    pairs = []
    for raw_pair in value.split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"Layer pair {pair!r} must use SOURCE:READ format, for example 10:20")
        source_raw, read_raw = pair.split(":", 1)
        pairs.append(LayerPair(source_layer=int(source_raw.strip()), read_layer=int(read_raw.strip())))
    return tuple(pairs)


def _hook_name_for_layer(layer: int, site: str) -> str:
    """Map a numeric layer/site pair to the TransformerLens residual hook name."""
    return build_hook_name(layer, site)


def _single_token_id(model: HookedTransformer, text: str) -> int:
    tokens = model.to_tokens(text, prepend_bos=False)[0]
    if len(tokens) != 1:
        decoded = [model.to_string(int(tok)) for tok in tokens]
        raise ValueError(f"Expected {text!r} to be one token, got {decoded}")
    return int(tokens[0])


def _load_model(model_name: str, device: torch.device) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(model_name)
    return model.to(device).eval()


class HFCausalLM:
    """Small adapter for Gemma models not yet exposed by TransformerLens."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        token = os.environ.get("HF_TOKEN")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, token=token)
        self.model = self.model.to(device).eval()
        self.device = device
        self.layers = self._get_layers()
        self.cfg = SimpleNamespace(n_layers=len(self.layers), d_model=int(self.model.config.hidden_size))

    def _get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise ValueError("Could not find decoder layers on the Hugging Face model.")

    def to_tokens(self, text: str, prepend_bos: bool = False) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=prepend_bos)
        return encoded.input_ids


def _single_token_id_hf(model: HFCausalLM, text: str) -> int:
    tokens = model.to_tokens(text, prepend_bos=False)[0]
    if len(tokens) != 1:
        decoded = [model.tokenizer.decode([int(tok)]) for tok in tokens]
        raise ValueError(f"Expected {text!r} to be one token, got {decoded}")
    return int(tokens[0])


@torch.no_grad()
def _hf_residual_at_last_token(
    model: HFCausalLM,
    prompt: str,
    layer: int,
    prepend_bos: bool,
) -> torch.Tensor:
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(model.device)
    last_idx = tokens.shape[1] - 1
    cache: dict[str, torch.Tensor] = {}

    def read_hook(module, inputs, output):  # type: ignore[no-untyped-def]
        hidden = output[0] if isinstance(output, tuple) else output
        cache["resid"] = hidden.detach().clone()
        return output

    handle = model.layers[layer].register_forward_hook(read_hook)
    try:
        model.model(tokens, use_cache=False)
    finally:
        handle.remove()
    return cache["resid"][0, last_idx, :].clone()


def _build_hf_vectors(
    model: HFCausalLM,
    source_layer: int,
    prompt_pairs: list[tuple[str, str]],
    prepend_bos: bool,
    model_name: str,
    behavior_name: str,
) -> tuple[torch.Tensor, str, str]:
    cache = VectorCache()
    cache_site = "hf_layer_output"
    cache_path = cache._get_cache_path(model_name, behavior_name, source_layer, cache_site)
    cached = cache.load(model_name, behavior_name, source_layer, cache_site, model.device)
    if cached is not None:
        return cached["bias"].to(model.device), "loaded:hf-layer-output", str(cache_path)

    diffs = []
    for positive_prompt, negative_prompt in prompt_pairs:
        pos = _hf_residual_at_last_token(model, positive_prompt, source_layer, prepend_bos)
        neg = _hf_residual_at_last_token(model, negative_prompt, source_layer, prepend_bos)
        diffs.append(pos - neg)

    bias_vec = unit_vector(torch.stack(diffs).mean(dim=0)).to(model.device)
    rand_vec = unit_vector(torch.randn_like(bias_vec))
    orth_seed = torch.randn_like(bias_vec)
    orth_vec = unit_vector(orth_seed - (orth_seed @ bias_vec) * bias_vec)
    cache.save({"bias": bias_vec, "random": rand_vec, "orth": orth_vec}, model_name, behavior_name, source_layer, cache_site)
    return bias_vec, "computed:hf-layer-output-diff-mean", str(cache_path)


@torch.no_grad()
def _hf_steered_logits_and_residual(
    model: HFCausalLM,
    prompt: str,
    steer_vec: torch.Tensor,
    alpha: float,
    source_layer: int,
    read_layer: int,
    prepend_bos: bool,
    steer_all_tokens: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(model.device)
    last_idx = tokens.shape[1] - 1
    cache: dict[str, torch.Tensor] = {}

    def steer_hook(module, inputs, output):  # type: ignore[no-untyped-def]
        hidden = output[0] if isinstance(output, tuple) else output
        vec = steer_vec.to(hidden.device, dtype=hidden.dtype)
        steered = hidden.clone()
        if steer_all_tokens:
            steered[:, :, :] = steered[:, :, :] + alpha * vec
        else:
            steered[:, last_idx, :] = steered[:, last_idx, :] + alpha * vec
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    def read_hook(module, inputs, output):  # type: ignore[no-untyped-def]
        hidden = output[0] if isinstance(output, tuple) else output
        cache["resid"] = hidden.detach().clone()
        return output

    handles = [
        model.layers[source_layer].register_forward_hook(steer_hook),
        model.layers[read_layer].register_forward_hook(read_hook),
    ]
    try:
        outputs = model.model(tokens, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    logits = outputs.logits[0, last_idx, :]
    return logits, cache["resid"][0, last_idx, :].detach().clone()


@torch.no_grad()
def _steered_logits_and_residual(
    model: HookedTransformer,
    prompt: str,
    steer_vec: torch.Tensor,
    alpha: float,
    source_hook: str,
    read_hook: str,
    source_layer: int,
    read_layer: int,
    prepend_bos: bool,
    device: torch.device,
    steer_all_tokens: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(device)
    last_idx = tokens.shape[1] - 1
    cache: dict[str, torch.Tensor] = {}

    def steer(act: torch.Tensor, hook) -> torch.Tensor:  # type: ignore[no-untyped-def]
        vec = steer_vec.to(act.device)
        if steer_all_tokens:
            act[:, :, :] = act[:, :, :] + alpha * vec
        else:
            act[:, last_idx, :] = act[:, last_idx, :] + alpha * vec
        return act

    def read(act: torch.Tensor, hook) -> torch.Tensor:  # type: ignore[no-untyped-def]
        cache["resid"] = act.detach().clone()
        return act

    model.run_with_hooks(
        tokens,
        return_type=None,
        stop_at_layer=max(source_layer, read_layer) + 1,
        fwd_hooks=[(source_hook, steer), (read_hook, read)],
    )

    read_resid = cache["resid"][:, last_idx : last_idx + 1, :]
    logits = model.unembed(model.ln_final(read_resid))[0, 0, :]
    return logits, read_resid[0, 0, :].detach().clone()


def _mean_metrics_for_alpha(
    model: HookedTransformer,
    prompts: list[str],
    steer_vec: torch.Tensor,
    alpha: float,
    baseline_residuals: list[torch.Tensor],
    choice1_id: int,
    choice2_id: int,
    cfg: GemmaSweepConfig,
    device: torch.device,
) -> dict[str, float]:
    source_hook = _hook_name_for_layer(cfg.source_layer, cfg.source_site)
    read_hook = _hook_name_for_layer(cfg.read_layer, cfg.read_site)

    behavior_scores = []
    logit_diffs = []
    residual_delta_norms = []

    for prompt, baseline_resid in zip(prompts, baseline_residuals):
        logits, resid = _steered_logits_and_residual(
            model=model,
            prompt=prompt,
            steer_vec=steer_vec,
            alpha=alpha,
            source_hook=source_hook,
            read_hook=read_hook,
            source_layer=cfg.source_layer,
            read_layer=cfg.read_layer,
            prepend_bos=cfg.prepend_bos,
            device=device,
            steer_all_tokens=cfg.steer_all_tokens,
        )

        pair_logits = logits[[choice1_id, choice2_id]]
        pair_probs = torch.softmax(pair_logits, dim=-1)
        behavior_scores.append(float(pair_probs[0].item()))
        logit_diffs.append(float((logits[choice1_id] - logits[choice2_id]).item()))
        residual_delta_norms.append(float((resid - baseline_resid).norm().item()))

    return {
        "alpha": float(alpha),
        "behavior_score": float(sum(behavior_scores) / len(behavior_scores)),
        "logit_difference": float(sum(logit_diffs) / len(logit_diffs)),
        "residual_delta_norm": float(sum(residual_delta_norms) / len(residual_delta_norms)),
    }


def _mean_hf_metrics_for_alpha(
    model: HFCausalLM,
    prompts: list[str],
    steer_vec: torch.Tensor,
    alpha: float,
    baseline_residuals: list[torch.Tensor],
    choice1_id: int,
    choice2_id: int,
    cfg: GemmaSweepConfig,
) -> dict[str, float]:
    behavior_scores = []
    logit_diffs = []
    residual_delta_norms = []

    for prompt, baseline_resid in zip(prompts, baseline_residuals):
        logits, resid = _hf_steered_logits_and_residual(
            model=model,
            prompt=prompt,
            steer_vec=steer_vec,
            alpha=alpha,
            source_layer=cfg.source_layer,
            read_layer=cfg.read_layer,
            prepend_bos=cfg.prepend_bos,
            steer_all_tokens=cfg.steer_all_tokens,
        )

        pair_logits = logits[[choice1_id, choice2_id]]
        pair_probs = torch.softmax(pair_logits.float(), dim=-1)
        behavior_scores.append(float(pair_probs[0].item()))
        logit_diffs.append(float((logits[choice1_id] - logits[choice2_id]).float().item()))
        residual_delta_norms.append(float((resid.float() - baseline_resid.float()).norm().item()))

    return {
        "alpha": float(alpha),
        "behavior_score": float(sum(behavior_scores) / len(behavior_scores)),
        "logit_difference": float(sum(logit_diffs) / len(logit_diffs)),
        "residual_delta_norm": float(sum(residual_delta_norms) / len(residual_delta_norms)),
    }


def _plot_series(results: list[dict[str, float]], key: str, ylabel: str, title: str, out_path: Path) -> None:
    alphas = [row["alpha"] for row in results]
    values = [row[key] for row in results]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, values, marker="o", linewidth=2.4)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel(r"Steering coefficient $\alpha$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _is_monotonic(values: Iterable[float]) -> bool:
    vals = list(values)
    if len(vals) < 2:
        return True
    return all(a <= b for a, b in zip(vals, vals[1:])) or all(a >= b for a, b in zip(vals, vals[1:]))


def _write_outputs(
    cfg: GemmaSweepConfig,
    output_dir: Path,
    results: list[dict[str, float]],
    vector_source: str,
    vector_cache_path: str,
    choice_tokens: dict[str, str],
    warnings: list[str],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "behavior_plot": str(output_dir / "alpha_vs_behavior_score.png"),
        "logit_plot": str(output_dir / "alpha_vs_logit_difference.png"),
        "residual_plot": str(output_dir / "alpha_vs_residual_delta_norm.png"),
        "json": str(output_dir / "alpha_sweep_results.json"),
        "csv": str(output_dir / "alpha_sweep_results.csv"),
        "layer_config": str(output_dir / "layer_config.json"),
        "summary": str(output_dir / "summary.md"),
    }

    title_prefix = (
        f"{cfg.model_name} | {cfg.behavior_name} | "
        f"source layer {cfg.source_layer} -> read layer {cfg.read_layer}"
    )
    _plot_series(
        results,
        "behavior_score",
        "Behavior score: P(Choice1 | A/B)",
        f"{title_prefix}\nAlpha vs Behavior Score",
        Path(paths["behavior_plot"]),
    )
    _plot_series(
        results,
        "logit_difference",
        "Logit difference: logit(A) - logit(B)",
        f"{title_prefix}\nAlpha vs Logit Difference",
        Path(paths["logit_plot"]),
    )
    _plot_series(
        results,
        "residual_delta_norm",
        "Residual delta norm vs alpha=0",
        f"{title_prefix}\nAlpha vs Residual Delta Norm",
        Path(paths["residual_plot"]),
    )

    payload = {
        "config": asdict(cfg),
        "vector_source": vector_source,
        "vector_cache_path": vector_cache_path,
        "choice_tokens": choice_tokens,
        "warnings": warnings,
        "results": results,
    }
    Path(paths["json"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(paths["layer_config"]).write_text(
        json.dumps(
            {
                "model_name": cfg.model_name,
                "behavior_name": cfg.behavior_name,
                "source_layer": cfg.source_layer,
                "read_layer": cfg.read_layer,
                "source_site": cfg.source_site,
                "read_site": cfg.read_site,
                "alpha_values": list(cfg.alpha_values),
                "max_train_prompts": cfg.max_train_prompts,
                "max_eval_prompts": cfg.max_eval_prompts,
                "vector_source": vector_source,
                "vector_cache_path": vector_cache_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with Path(paths["csv"]).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "behavior_score", "logit_difference", "residual_delta_norm"])
        writer.writeheader()
        writer.writerows(results)

    behavior_monotonic = _is_monotonic(row["behavior_score"] for row in results)
    logit_monotonic = _is_monotonic(row["logit_difference"] for row in results)
    residual_monotonic = _is_monotonic(row["residual_delta_norm"] for row in results)
    max_resid = max(row["residual_delta_norm"] for row in results)
    oversteering_note = "No obvious collapse signal from residual norm alone."
    if max_resid > 100:
        oversteering_note = "Residual delta norm is large; inspect generations/logits for oversteering."

    report = f"""# Gemma Alpha Sweep Report

## Setup
- Model: `{cfg.model_name}`
- Behavior/concept: `{cfg.behavior_name}` forced-choice behavior
- Source layer/site: `{cfg.source_layer}:{cfg.source_site}`
- Read layer/site: `{cfg.read_layer}:{cfg.read_site}`
- Alpha values: `{list(cfg.alpha_values)}`
- Max train prompts: `{cfg.max_train_prompts}`
- Max eval prompts: `{cfg.max_eval_prompts}`
- Steering vector: `{vector_source}`
- Steering vector cache path: `{vector_cache_path}`
- Choice tokens: `{choice_tokens}`

## Results
- Behavior score monotonic: `{behavior_monotonic}`
- Logit difference monotonic: `{logit_monotonic}`
- Residual norm monotonic: `{residual_monotonic}`
- Oversteering/collapse note: {oversteering_note}

## What Worked
- Reused the existing CBMAS JSON behavior datasets.
- Reused the existing dense diff-mean steering vector construction and cache.
- Captured read-layer residual state and measured delta from alpha=0.

## Known Limitations / Warnings
{chr(10).join(f"- {warning}" for warning in warnings) if warnings else "- No warnings recorded during script construction."}

## Saved Files
- `{paths["behavior_plot"]}`
- `{paths["logit_plot"]}`
- `{paths["residual_plot"]}`
- `{paths["json"]}`
- `{paths["csv"]}`
- `{paths["layer_config"]}`
"""
    Path(paths["summary"]).write_text(report, encoding="utf-8")
    return paths


def _validate_layers(cfg: GemmaSweepConfig, n_layers: int, warnings: list[str]) -> None:
    if cfg.source_layer < 0 or cfg.source_layer >= n_layers:
        raise ValueError(f"source_layer={cfg.source_layer} is outside model layer range 0..{n_layers - 1}")
    if cfg.read_layer < 0 or cfg.read_layer >= n_layers:
        raise ValueError(f"read_layer={cfg.read_layer} is outside model layer range 0..{n_layers - 1}")
    if cfg.read_layer <= cfg.source_layer:
        warnings.append("Read layer is not later than source layer; this is allowed but may be harder to interpret.")


def _load_behavior_data(cfg: GemmaSweepConfig) -> tuple[list[tuple[str, str]], list[str]]:
    train_pairs = load_train_dataset(cfg.behavior_name)[: cfg.max_train_prompts]
    eval_prompts = load_test_dataset(cfg.behavior_name)[: cfg.max_eval_prompts]
    if not train_pairs:
        raise ValueError(f"No training prompts found for behavior {cfg.behavior_name!r}")
    if not eval_prompts:
        raise ValueError(f"No eval prompts found for behavior {cfg.behavior_name!r}")
    return train_pairs, eval_prompts


def _plot_comparison(
    summaries: list[dict[str, object]],
    key: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for summary in summaries:
        results = summary["results"]
        assert isinstance(results, list)
        alphas = [row["alpha"] for row in results]
        values = [row[key] for row in results]
        ax.plot(alphas, values, marker="o", linewidth=2.0, label=str(summary["layer_label"]))
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel(r"Steering coefficient $\alpha$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_combined_results(run_root: Path, summaries: list[dict[str, object]]) -> str:
    out_path = run_root / "combined_results.csv"
    fieldnames = [
        "layer_label",
        "source_layer",
        "read_layer",
        "alpha",
        "behavior_score",
        "logit_difference",
        "residual_delta_norm",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            results = summary["results"]
            assert isinstance(results, list)
            for row in results:
                writer.writerow(
                    {
                        "layer_label": summary["layer_label"],
                        "source_layer": summary["source_layer"],
                        "read_layer": summary["read_layer"],
                        **row,
                    }
                )
    return str(out_path)


def _trend_note(results: list[dict[str, float]], key: str) -> str:
    if not results:
        return "no results"
    start = results[0][key]
    end = results[-1][key]
    if end > start:
        direction = "increased"
    elif end < start:
        direction = "decreased"
    else:
        direction = "stayed flat"
    return f"{direction} from {start:.4f} to {end:.4f}"


def _write_run_summary(
    run_root: Path,
    cfg: GemmaSweepConfig,
    run_id: str,
    summaries: list[dict[str, object]],
    warnings: list[str],
    comparison_paths: dict[str, str],
    combined_results_path: str | None,
) -> str:
    lines = [
        "# Gemma Layer Sweep Run Summary",
        "",
        "## Run",
        f"- Run ID: `{run_id}`",
        f"- Model: `{cfg.model_name}`",
        f"- Behavior/concept: `{cfg.behavior_name}`",
        f"- Alpha values: `{list(cfg.alpha_values)}`",
        f"- Max train prompts: `{cfg.max_train_prompts}`",
        f"- Max eval prompts: `{cfg.max_eval_prompts}`",
        "",
        "## Layer Pairs",
    ]

    for summary in summaries:
        results = summary["results"]
        assert isinstance(results, list)
        behavior_note = _trend_note(results, "behavior_score")
        logit_note = _trend_note(results, "logit_difference")
        residual_note = _trend_note(results, "residual_delta_norm")
        max_residual = max(row["residual_delta_norm"] for row in results)
        collapse_note = "No obvious oversteering/collapse signal from residual norm alone."
        if max_residual > 100:
            collapse_note = "Residual delta norm is large; inspect generations/logits for oversteering."
        lines.extend(
            [
                f"- `{summary['layer_label']}`",
                f"  Source/read: `{summary['source_layer']} -> {summary['read_layer']}`",
                f"  Behavior score: {behavior_note}",
                f"  Logit difference: {logit_note}",
                f"  Residual delta norm: {residual_note}",
                f"  Oversteering/collapse: {collapse_note}",
                f"  Summary: `{summary['paths']['summary']}`",
            ]
        )

    lines.extend(["", "## Comparison Files"])
    if comparison_paths:
        lines.extend(f"- `{path}`" for path in comparison_paths.values())
    else:
        lines.append("- No comparison plots were created because only one layer pair was run.")
    if combined_results_path:
        lines.append(f"- `{combined_results_path}`")

    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in sorted(set(warnings))) if warnings else lines.append("- None")

    out_path = run_root / "run_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out_path)


def _run_transformer_lens_sweep(
    cfg: GemmaSweepConfig,
    warnings: list[str],
    device: torch.device,
    model: HookedTransformer,
) -> dict[str, object]:
    _validate_layers(cfg, int(model.cfg.n_layers), warnings)
    train_pairs, eval_prompts = _load_behavior_data(cfg)

    cache = VectorCache()
    vector_cache_path = str(cache._get_cache_path(cfg.model_name, cfg.behavior_name, cfg.source_layer, cfg.source_site))
    vector_status = "loaded" if Path(vector_cache_path).exists() else "computed"
    vectors = build_vectors(
        model=model,
        inj_layer=cfg.source_layer,
        prompt_pairs=train_pairs,
        prepend_bos=cfg.prepend_bos,
        device=device,
        inject_site=cfg.source_site,
        model_name=cfg.model_name,
        dataset=cfg.behavior_name,
    )
    steer_vec = vectors["bias"]
    vector_source = (
        f"transformer_lens:{vector_status}:diff-mean:"
        f"{cfg.model_name}:{cfg.behavior_name}:L{cfg.source_layer}:{cfg.source_site}"
    )

    choice1_id = _single_token_id(model, "A")
    choice2_id = _single_token_id(model, "B")
    choice_tokens = {
        "choice1_text": "A",
        "choice1_id": str(choice1_id),
        "choice1_decoded": model.tokenizer.decode([choice1_id]),
        "choice2_text": "B",
        "choice2_id": str(choice2_id),
        "choice2_decoded": model.tokenizer.decode([choice2_id]),
    }

    baseline_residuals = []
    for prompt in eval_prompts:
        _, baseline_resid = _steered_logits_and_residual(
            model=model,
            prompt=prompt,
            steer_vec=steer_vec,
            alpha=0.0,
            source_hook=_hook_name_for_layer(cfg.source_layer, cfg.source_site),
            read_hook=_hook_name_for_layer(cfg.read_layer, cfg.read_site),
            source_layer=cfg.source_layer,
            read_layer=cfg.read_layer,
            prepend_bos=cfg.prepend_bos,
            device=device,
            steer_all_tokens=cfg.steer_all_tokens,
        )
        baseline_residuals.append(baseline_resid)

    results = [
        _mean_metrics_for_alpha(
            model=model,
            prompts=eval_prompts,
            steer_vec=steer_vec,
            alpha=alpha,
            baseline_residuals=baseline_residuals,
            choice1_id=choice1_id,
            choice2_id=choice2_id,
            cfg=cfg,
            device=device,
        )
        for alpha in cfg.alpha_values
    ]

    out_root = Path(cfg.out_dir)
    paths = _write_outputs(cfg, out_root, results, vector_source, vector_cache_path, choice_tokens, warnings)

    summary = {
        "model": cfg.model_name,
        "backend": "transformer_lens",
        "behavior": cfg.behavior_name,
        "source_layer": cfg.source_layer,
        "read_layer": cfg.read_layer,
        "alpha_values": list(cfg.alpha_values),
        "num_train_prompts": len(train_pairs),
        "num_eval_prompts": len(eval_prompts),
        "vector_source": vector_source,
        "vector_cache_path": vector_cache_path,
        "results": results,
        "paths": paths,
        "warnings": warnings,
    }
    print(json.dumps(summary, indent=2))
    return summary


def _run_huggingface_sweep(
    cfg: GemmaSweepConfig,
    warnings: list[str],
    device: torch.device,
    fallback_reason: str,
    model: HFCausalLM | None = None,
) -> dict[str, object]:
    warnings.append(f"TransformerLens backend unavailable for this model; using Hugging Face backend. Reason: {fallback_reason}")
    warnings.append("HF backend steers decoder layer outputs, not TransformerLens residual hook sites.")
    if model is None:
        model = HFCausalLM(cfg.model_name, device)
    _validate_layers(cfg, int(model.cfg.n_layers), warnings)
    train_pairs, eval_prompts = _load_behavior_data(cfg)

    steer_vec, vector_status, vector_cache_path = _build_hf_vectors(
        model=model,
        source_layer=cfg.source_layer,
        prompt_pairs=train_pairs,
        prepend_bos=cfg.prepend_bos,
        model_name=cfg.model_name,
        behavior_name=cfg.behavior_name,
    )
    vector_source = f"huggingface:{vector_status}:{cfg.model_name}:{cfg.behavior_name}:L{cfg.source_layer}"

    choice1_id = _single_token_id_hf(model, "A")
    choice2_id = _single_token_id_hf(model, "B")
    choice_tokens = {
        "choice1_text": "A",
        "choice1_id": str(choice1_id),
        "choice1_decoded": model.tokenizer.decode([choice1_id]),
        "choice2_text": "B",
        "choice2_id": str(choice2_id),
        "choice2_decoded": model.tokenizer.decode([choice2_id]),
    }

    baseline_residuals = []
    for prompt in eval_prompts:
        _, baseline_resid = _hf_steered_logits_and_residual(
            model=model,
            prompt=prompt,
            steer_vec=steer_vec,
            alpha=0.0,
            source_layer=cfg.source_layer,
            read_layer=cfg.read_layer,
            prepend_bos=cfg.prepend_bos,
            steer_all_tokens=cfg.steer_all_tokens,
        )
        baseline_residuals.append(baseline_resid)

    results = [
        _mean_hf_metrics_for_alpha(
            model=model,
            prompts=eval_prompts,
            steer_vec=steer_vec,
            alpha=alpha,
            baseline_residuals=baseline_residuals,
            choice1_id=choice1_id,
            choice2_id=choice2_id,
            cfg=cfg,
        )
        for alpha in cfg.alpha_values
    ]

    out_root = Path(cfg.out_dir)
    paths = _write_outputs(cfg, out_root, results, vector_source, vector_cache_path, choice_tokens, warnings)

    summary = {
        "model": cfg.model_name,
        "backend": "huggingface",
        "behavior": cfg.behavior_name,
        "source_layer": cfg.source_layer,
        "read_layer": cfg.read_layer,
        "alpha_values": list(cfg.alpha_values),
        "num_train_prompts": len(train_pairs),
        "num_eval_prompts": len(eval_prompts),
        "vector_source": vector_source,
        "vector_cache_path": vector_cache_path,
        "results": results,
        "paths": paths,
        "warnings": warnings,
    }
    print(json.dumps(summary, indent=2))
    return summary


def run_sweep(cfg: GemmaSweepConfig) -> dict[str, object]:
    base_warnings = [
        "Behavior score is a diagnostic placeholder: normalized probability of Choice1 over Choice1/Choice2.",
        "SAE availability is not checked here; this experiment uses dense residual steering only.",
        "Gemma access may require accepting the Hugging Face model terms and providing a valid token.",
    ]
    configure_determinism(cfg.seed)
    device = get_device()
    run_id = cfg.run_id or _make_run_id()
    layer_pairs = cfg.layer_pairs or (LayerPair(cfg.source_layer, cfg.read_layer),)
    run_root = Path(cfg.out_dir) / _clean_model_name(cfg.model_name) / cfg.behavior_name / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        **asdict(cfg),
        "run_id": run_id,
        "layer_pairs": [asdict(pair) for pair in layer_pairs],
        "output_dir": str(run_root),
    }
    (run_root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": run_id,
                "model": cfg.model_name,
                "behavior": cfg.behavior_name,
                "alpha_values": list(cfg.alpha_values),
                "max_train_prompts": cfg.max_train_prompts,
                "max_eval_prompts": cfg.max_eval_prompts,
                "layer_pairs": [asdict(pair) for pair in layer_pairs],
                "output_dir": str(run_root),
            },
            indent=2,
        )
    )

    try:
        model = _load_model(cfg.model_name, device)
    except ValueError as exc:
        if "not found" not in str(exc):
            raise
        fallback_reason = str(exc).splitlines()[0]
        hf_model = HFCausalLM(cfg.model_name, device)
        backend = "huggingface"
        summaries = []
        all_warnings = list(base_warnings)
        for pair in layer_pairs:
            pair_dir = run_root / _layer_label(pair.source_layer, pair.read_layer)
            pair_cfg = replace(
                cfg,
                source_layer=pair.source_layer,
                read_layer=pair.read_layer,
                out_dir=str(pair_dir),
                run_id=run_id,
                layer_pairs=tuple(),
            )
            pair_warnings = list(base_warnings)
            print(
                json.dumps(
                    {
                        "source_layer": pair.source_layer,
                        "read_layer": pair.read_layer,
                        "output_dir": str(pair_dir),
                    },
                    indent=2,
                )
            )
            summary = _run_huggingface_sweep(pair_cfg, pair_warnings, device, fallback_reason, hf_model)
            summary["layer_label"] = _comparison_label(pair.source_layer, pair.read_layer)
            summaries.append(summary)
            all_warnings.extend(pair_warnings)
    else:
        backend = "transformer_lens"
        summaries = []
        all_warnings = list(base_warnings)
        for pair in layer_pairs:
            pair_dir = run_root / _layer_label(pair.source_layer, pair.read_layer)
            pair_cfg = replace(
                cfg,
                source_layer=pair.source_layer,
                read_layer=pair.read_layer,
                out_dir=str(pair_dir),
                run_id=run_id,
                layer_pairs=tuple(),
            )
            pair_warnings = list(base_warnings)
            print(
                json.dumps(
                    {
                        "source_layer": pair.source_layer,
                        "read_layer": pair.read_layer,
                        "output_dir": str(pair_dir),
                    },
                    indent=2,
                )
            )
            summary = _run_transformer_lens_sweep(pair_cfg, pair_warnings, device, model)
            summary["layer_label"] = _comparison_label(pair.source_layer, pair.read_layer)
            summaries.append(summary)
            all_warnings.extend(pair_warnings)

    comparison_paths: dict[str, str] = {}
    combined_results_path: str | None = None
    if len(summaries) > 1:
        comparison_specs = [
            (
                "behavior_score",
                "Behavior score: P(Choice1 | A/B)",
                "compare_alpha_vs_behavior_score.png",
                "Layer Comparison: Alpha vs Behavior Score",
            ),
            (
                "logit_difference",
                "Logit difference: logit(A) - logit(B)",
                "compare_alpha_vs_logit_difference.png",
                "Layer Comparison: Alpha vs Logit Difference",
            ),
            (
                "residual_delta_norm",
                "Residual delta norm vs alpha=0",
                "compare_alpha_vs_residual_delta_norm.png",
                "Layer Comparison: Alpha vs Residual Delta Norm",
            ),
        ]
        for key, ylabel, filename, title in comparison_specs:
            out_path = run_root / filename
            _plot_comparison(summaries, key, ylabel, title, out_path)
            comparison_paths[key] = str(out_path)
        combined_results_path = _write_combined_results(run_root, summaries)

    run_summary_path = _write_run_summary(
        run_root=run_root,
        cfg=cfg,
        run_id=run_id,
        summaries=summaries,
        warnings=all_warnings,
        comparison_paths=comparison_paths,
        combined_results_path=combined_results_path,
    )

    final_summary = {
        "run_id": run_id,
        "backend": backend,
        "model": cfg.model_name,
        "behavior": cfg.behavior_name,
        "output_dir": str(run_root),
        "run_config": str(run_root / "run_config.json"),
        "run_summary": run_summary_path,
        "combined_results": combined_results_path,
        "comparison_plots": comparison_paths,
        "layer_summaries": summaries,
    }
    print(json.dumps(final_summary, indent=2))
    return final_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small Gemma alpha sweep diagnostic.")
    parser.add_argument("--model-name", "--model", default="google/gemma-2-2b-it")
    parser.add_argument("--behavior-name", "--behavior", default="reassurance")
    parser.add_argument("--source-layer", type=int, default=12)
    parser.add_argument("--read-layer", type=int, default=20)
    parser.add_argument("--layer-pairs", default="", help='Comma-separated SOURCE:READ pairs, for example "8:16,10:20,12:24".')
    parser.add_argument("--source-site", default="hook_resid_mid")
    parser.add_argument("--read-site", default="hook_resid_post")
    parser.add_argument("--alpha-values", default=",".join(str(x).removesuffix(".0") for x in DEFAULT_ALPHAS))
    parser.add_argument("--alphas", nargs="+", type=float, default=None, help="Space-separated alpha values, for example --alphas -4 -2 -1 0 1 2 4.")
    parser.add_argument("--max-train-prompts", type=int, default=40)
    parser.add_argument("--max-eval-prompts", type=int, default=30)
    parser.add_argument("--num-prompts", type=int, default=None, help="Shortcut that sets both max train and max eval prompts.")
    parser.add_argument("--out-dir", default="graphs/gemma_alpha_sweep")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-prepend-bos", dest="prepend_bos", action="store_false")
    parser.set_defaults(prepend_bos=True)
    parser.add_argument("--steer-last-token-only", dest="steer_all_tokens", action="store_false")
    parser.set_defaults(steer_all_tokens=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    max_train_prompts = args.num_prompts if args.num_prompts is not None else args.max_train_prompts
    max_eval_prompts = args.num_prompts if args.num_prompts is not None else args.max_eval_prompts
    alpha_values = tuple(args.alphas) if args.alphas is not None else _parse_alpha_values(args.alpha_values)
    cfg = GemmaSweepConfig(
        model_name=args.model_name,
        behavior_name=args.behavior_name,
        source_layer=args.source_layer,
        read_layer=args.read_layer,
        source_site=args.source_site,
        read_site=args.read_site,
        alpha_values=alpha_values,
        max_train_prompts=max_train_prompts,
        max_eval_prompts=max_eval_prompts,
        prepend_bos=args.prepend_bos,
        steer_all_tokens=args.steer_all_tokens,
        seed=args.seed,
        out_dir=args.out_dir,
        run_id=args.run_id,
        layer_pairs=_parse_layer_pairs(args.layer_pairs),
    )
    run_sweep(cfg)


if __name__ == "__main__":
    main()
