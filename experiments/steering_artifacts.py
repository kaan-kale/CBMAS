"""Capture hidden-state artifacts for steered vs unsteered Gemma runs.

This is intentionally separate from the alpha-sweep experiment. It gives us a
small debug pipeline for saving read-layer activations first, while leaving SAE
encoding optional and non-blocking.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from BRC_Experiment.Modularized.cache import VectorCache
from BRC_Experiment.Modularized.data import load_test_dataset, load_train_dataset
from BRC_Experiment.Modularized.utils import configure_determinism, get_device, unit_vector


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cbmas_matplotlib"))


@dataclass
class SteeringArtifactConfig:
    model_name: str = "google/gemma-2-2b"
    behavior_name: str = "reassurance"
    source_layer: int = 8
    read_layer: int = 16
    alpha: float = 1.0
    max_train_prompts: int = 40
    max_eval_prompts: int = 5
    prepend_bos: bool = True
    steer_all_tokens: bool = True
    seed: int = 42
    out_dir: str = "steering_runs"
    run_id: str = ""
    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    sae_id: str = "layer_16/width_16k/canonical"


class HFCausalLM:
    """Minimal Hugging Face adapter for Gemma hidden-state collection."""

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


def _make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{uuid.uuid4().hex[:4]}"


def _clean_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


def _validate_layers(cfg: SteeringArtifactConfig, n_layers: int) -> None:
    if cfg.source_layer < 0 or cfg.source_layer >= n_layers:
        raise ValueError(f"source_layer={cfg.source_layer} is outside model layer range 0..{n_layers - 1}")
    if cfg.read_layer < 0 or cfg.read_layer >= n_layers:
        raise ValueError(f"read_layer={cfg.read_layer} is outside model layer range 0..{n_layers - 1}")


@torch.no_grad()
def collect_read_layer_hidden_state(
    model: HFCausalLM,
    prompt: str,
    read_layer: int,
    prepend_bos: bool,
) -> torch.Tensor:
    """Collect the full read-layer hidden state without steering."""
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(model.device)
    cache: dict[str, torch.Tensor] = {}

    def read_hook(module, inputs, output):  # type: ignore[no-untyped-def]
        hidden = output[0] if isinstance(output, tuple) else output
        cache["hidden"] = hidden.detach().clone()
        return output

    handle = model.layers[read_layer].register_forward_hook(read_hook)
    try:
        model.model(tokens, use_cache=False)
    finally:
        handle.remove()

    return cache["hidden"][0].detach().cpu()


@torch.no_grad()
def run_steered_forward(
    model: HFCausalLM,
    prompt: str,
    steer_vec: torch.Tensor,
    alpha: float,
    source_layer: int,
    read_layer: int,
    prepend_bos: bool,
    steer_all_tokens: bool,
) -> torch.Tensor:
    """Run one steered forward pass and return the full read-layer hidden state."""
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
        cache["hidden"] = hidden.detach().clone()
        return output

    handles = [
        model.layers[source_layer].register_forward_hook(steer_hook),
        model.layers[read_layer].register_forward_hook(read_hook),
    ]
    try:
        model.model(tokens, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    return cache["hidden"][0].detach().cpu()


@torch.no_grad()
def _residual_at_last_token(
    model: HFCausalLM,
    prompt: str,
    layer: int,
    prepend_bos: bool,
) -> torch.Tensor:
    hidden = collect_read_layer_hidden_state(model, prompt, layer, prepend_bos)
    return hidden[-1, :].to(model.device)


def _build_hf_steering_vector(
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
        return cached["bias"].to(model.device), "loaded", str(cache_path)

    diffs = []
    for positive_prompt, negative_prompt in prompt_pairs:
        pos = _residual_at_last_token(model, positive_prompt, source_layer, prepend_bos)
        neg = _residual_at_last_token(model, negative_prompt, source_layer, prepend_bos)
        diffs.append(pos - neg)

    bias_vec = unit_vector(torch.stack(diffs).mean(dim=0)).to(model.device)
    rand_vec = unit_vector(torch.randn_like(bias_vec))
    orth_seed = torch.randn_like(bias_vec)
    orth_vec = unit_vector(orth_seed - (orth_seed @ bias_vec) * bias_vec)
    cache.save({"bias": bias_vec, "random": rand_vec, "orth": orth_vec}, model_name, behavior_name, source_layer, cache_site)
    return bias_vec, "computed", str(cache_path)


def _load_optional_sae(cfg: SteeringArtifactConfig, device: torch.device) -> tuple[Any | None, dict[str, Any]]:
    if not cfg.sae_release or not cfg.sae_id:
        return None, {
            "enabled": False,
            "reason": "No SAE release/id was provided.",
            "release": cfg.sae_release,
            "sae_id": cfg.sae_id,
        }

    try:
        from sae_lens import SAE  # type: ignore
    except ImportError:
        return None, {
            "enabled": False,
            "reason": "sae_lens is not installed in this environment.",
            "release": cfg.sae_release,
            "sae_id": cfg.sae_id,
        }

    loaded = SAE.from_pretrained(release=cfg.sae_release, sae_id=cfg.sae_id, device=str(device))
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    if hasattr(sae, "eval"):
        sae.eval()
    return sae, {
        "enabled": True,
        "release": cfg.sae_release,
        "sae_id": cfg.sae_id,
    }


@torch.no_grad()
def encode_with_sae(sae: Any, hidden_states: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Encode real read-layer activations, never the steering vector itself."""
    encoded = sae.encode(hidden_states.unsqueeze(0).to(device))
    features = encoded[0] if isinstance(encoded, tuple) else encoded
    return features[0].detach().cpu()


def compare_sae_features(unsteered: torch.Tensor, steered: torch.Tensor, top_k: int = 25) -> dict[str, Any]:
    delta = steered - unsteered
    mean_delta = delta.mean(dim=0)
    abs_mean_delta = mean_delta.abs()
    top_values, top_indices = torch.topk(abs_mean_delta, k=min(top_k, abs_mean_delta.numel()))
    return {
        "feature_dim": int(unsteered.shape[-1]),
        "unsteered_l0_mean": float((unsteered > 0).sum(dim=-1).float().mean().item()),
        "steered_l0_mean": float((steered > 0).sum(dim=-1).float().mean().item()),
        "unsteered_norm_mean": float(unsteered.norm(dim=-1).mean().item()),
        "steered_norm_mean": float(steered.norm(dim=-1).mean().item()),
        "delta_norm_mean": float(delta.norm(dim=-1).mean().item()),
        "top_changed_features": [
            {
                "feature": int(idx.item()),
                "abs_mean_delta": float(value.item()),
                "mean_delta": float(mean_delta[idx].item()),
            }
            for idx, value in zip(top_indices, top_values)
        ],
    }


def _save_prompt_artifacts(
    run_root: Path,
    prompt_id: int,
    unsteered_hidden: torch.Tensor,
    steered_hidden: torch.Tensor,
    sae_unsteered: torch.Tensor | None,
    sae_steered: torch.Tensor | None,
) -> dict[str, str]:
    hidden_dir = run_root / "hidden_states"
    sae_dir = run_root / "sae_features"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    sae_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "unsteered_hidden": str(hidden_dir / f"prompt_{prompt_id:04d}_unsteered.pt"),
        "steered_hidden": str(hidden_dir / f"prompt_{prompt_id:04d}_steered.pt"),
    }
    torch.save(unsteered_hidden, paths["unsteered_hidden"])
    torch.save(steered_hidden, paths["steered_hidden"])

    if sae_unsteered is not None and sae_steered is not None:
        paths["unsteered_sae"] = str(sae_dir / f"prompt_{prompt_id:04d}_unsteered.pt")
        paths["steered_sae"] = str(sae_dir / f"prompt_{prompt_id:04d}_steered.pt")
        torch.save(sae_unsteered, paths["unsteered_sae"])
        torch.save(sae_steered, paths["steered_sae"])

    return paths


def run_artifact_capture(cfg: SteeringArtifactConfig) -> dict[str, Any]:
    configure_determinism(cfg.seed)
    device = get_device()
    run_id = cfg.run_id or _make_run_id()
    run_root = Path(cfg.out_dir) / _clean_model_name(cfg.model_name) / cfg.behavior_name / run_id
    summaries_dir = run_root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    model = HFCausalLM(cfg.model_name, device)
    _validate_layers(cfg, int(model.cfg.n_layers))

    train_pairs = load_train_dataset(cfg.behavior_name)[: cfg.max_train_prompts]
    eval_prompts = load_test_dataset(cfg.behavior_name)[: cfg.max_eval_prompts]
    if not train_pairs:
        raise ValueError(f"No training prompts found for behavior {cfg.behavior_name!r}")
    if not eval_prompts:
        raise ValueError(f"No eval prompts found for behavior {cfg.behavior_name!r}")

    steer_vec, vector_status, vector_cache_path = _build_hf_steering_vector(
        model=model,
        source_layer=cfg.source_layer,
        prompt_pairs=train_pairs,
        prepend_bos=cfg.prepend_bos,
        model_name=cfg.model_name,
        behavior_name=cfg.behavior_name,
    )
    sae, sae_status = _load_optional_sae(cfg, device)

    prompt_rows = []
    feature_summaries = []
    for prompt_id, prompt in enumerate(eval_prompts):
        unsteered_hidden = collect_read_layer_hidden_state(model, prompt, cfg.read_layer, cfg.prepend_bos)
        steered_hidden = run_steered_forward(
            model=model,
            prompt=prompt,
            steer_vec=steer_vec,
            alpha=cfg.alpha,
            source_layer=cfg.source_layer,
            read_layer=cfg.read_layer,
            prepend_bos=cfg.prepend_bos,
            steer_all_tokens=cfg.steer_all_tokens,
        )

        sae_unsteered = None
        sae_steered = None
        feature_summary = None
        if sae is not None:
            sae_unsteered = encode_with_sae(sae, unsteered_hidden, device)
            sae_steered = encode_with_sae(sae, steered_hidden, device)
            feature_summary = compare_sae_features(sae_unsteered, sae_steered)
            feature_summary["prompt_id"] = prompt_id
            feature_summaries.append(feature_summary)

        paths = _save_prompt_artifacts(run_root, prompt_id, unsteered_hidden, steered_hidden, sae_unsteered, sae_steered)
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "seq_len": int(unsteered_hidden.shape[0]),
                "hidden_dim": int(unsteered_hidden.shape[1]),
                "hidden_delta_norm_last_token": float((steered_hidden[-1] - unsteered_hidden[-1]).norm().item()),
                "hidden_delta_norm_mean": float((steered_hidden - unsteered_hidden).norm(dim=-1).mean().item()),
                "paths": paths,
            }
        )

    prompt_summary_path = summaries_dir / "prompt_hidden_state_summary.csv"
    with prompt_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_id",
                "seq_len",
                "hidden_dim",
                "hidden_delta_norm_last_token",
                "hidden_delta_norm_mean",
            ],
        )
        writer.writeheader()
        for row in prompt_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    feature_summary_path = summaries_dir / "sae_feature_summary.json"
    feature_summary_path.write_text(json.dumps(feature_summaries, indent=2), encoding="utf-8")

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_name": cfg.model_name,
        "behavior_name": cfg.behavior_name,
        "prompt_ids": [row["prompt_id"] for row in prompt_rows],
        "source_layer": cfg.source_layer,
        "read_layer": cfg.read_layer,
        "alpha": cfg.alpha,
        "vector_status": vector_status,
        "vector_name": "bias",
        "vector_cache_path": vector_cache_path,
        "sae": sae_status,
        "used_transformer_lens": False,
        "used_saelens": bool(sae_status["enabled"]),
        "used_neuronpedia": False,
        "run_settings": asdict(cfg),
        "artifact_paths": {
            "hidden_states": str(run_root / "hidden_states"),
            "sae_features": str(run_root / "sae_features"),
            "summaries": str(summaries_dir),
            "prompt_summary": str(prompt_summary_path),
            "feature_summary": str(feature_summary_path),
        },
        "prompts": prompt_rows,
        "methodological_note": "SAE encoding is applied to real read-layer activations, not to the dense steering vector.",
    }
    metadata_path = run_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if feature_summaries:
        top = feature_summaries[0]["top_changed_features"][:10]
        print(json.dumps({"top_changed_features_prompt_0": top}, indent=2))
    else:
        print(f"SAE features were not saved: {sae_status['reason']}")

    final_summary = {
        "run_id": run_id,
        "output_dir": str(run_root),
        "metadata": str(metadata_path),
        "prompt_summary": str(prompt_summary_path),
        "feature_summary": str(feature_summary_path),
        "sae_status": sae_status,
    }
    print(json.dumps(final_summary, indent=2))
    return final_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save steered vs unsteered read-layer artifacts.")
    parser.add_argument("--model-name", "--model", default="google/gemma-2-2b")
    parser.add_argument("--behavior-name", "--behavior", default="reassurance")
    parser.add_argument("--source-layer", type=int, default=8)
    parser.add_argument("--read-layer", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-train-prompts", type=int, default=40)
    parser.add_argument("--max-eval-prompts", "--num-prompts", type=int, default=5)
    parser.add_argument("--out-dir", default="steering_runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    parser.add_argument("--sae-id", default="layer_16/width_16k/canonical")
    parser.add_argument("--no-prepend-bos", dest="prepend_bos", action="store_false")
    parser.set_defaults(prepend_bos=True)
    parser.add_argument("--steer-last-token-only", dest="steer_all_tokens", action="store_false")
    parser.set_defaults(steer_all_tokens=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = SteeringArtifactConfig(
        model_name=args.model_name,
        behavior_name=args.behavior_name,
        source_layer=args.source_layer,
        read_layer=args.read_layer,
        alpha=args.alpha,
        max_train_prompts=args.max_train_prompts,
        max_eval_prompts=args.max_eval_prompts,
        prepend_bos=args.prepend_bos,
        steer_all_tokens=args.steer_all_tokens,
        seed=args.seed,
        out_dir=args.out_dir,
        run_id=args.run_id,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
    )
    run_artifact_capture(cfg)


if __name__ == "__main__":
    main()
