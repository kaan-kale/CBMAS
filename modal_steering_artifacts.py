"""Thin Modal wrapper for steered/unsteered artifact capture."""

from __future__ import annotations

import os
from pathlib import Path

import modal


APP_NAME = "cbmas-steering-artifacts"
REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = "/root"

artifacts_vol = modal.Volume.from_name("cbmas-steering-artifacts", create_if_missing=True)
cache_vol = modal.Volume.from_name("cbmas-cache", create_if_missing=True)

hf_secret = (
    modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})
    if modal.is_local() and os.environ.get("HF_TOKEN")
    else modal.Secret.from_dict({})
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.26,<2.0",
        "torch>=2.6,<3",
        "transformers>=4.44,<5",
        "datasets>=2.20,<3",
        "sentencepiece>=0.2,<0.3",
        "sae-lens>=6,<7",
    )
    .add_local_python_source("BRC_Experiment")
    .add_local_python_source("experiments")
    .add_local_dir(str(REPO_ROOT / "data"), remote_path=f"{PROJECT_ROOT}/data")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu="L4",
    timeout=60 * 60,
    volumes={
        f"{PROJECT_ROOT}/steering_runs": artifacts_vol,
        f"{PROJECT_ROOT}/cache": cache_vol,
    },
    secrets=[hf_secret],
)
def run_steering_artifacts(argv: list[str]) -> str:
    os.chdir(PROJECT_ROOT)

    from experiments.steering_artifacts import main

    main(argv)
    artifacts_vol.commit()
    cache_vol.commit()
    return "Finished steering artifact capture"


@app.local_entrypoint()
def main(
    model_name: str = "google/gemma-2-2b",
    behavior_name: str = "reassurance",
    source_layer: int = 8,
    read_layer: int = 16,
    alpha: float = 1.0,
    max_train_prompts: int = 40,
    max_eval_prompts: int = 5,
    seed: int = 42,
    run_id: str = "",
    sae_release: str = "gemma-scope-2b-pt-res-canonical",
    sae_id: str = "layer_16/width_16k/canonical",
    prepend_bos: bool = True,
    steer_all_tokens: bool = True,
) -> None:
    argv = [
        "--model-name",
        model_name,
        "--behavior-name",
        behavior_name,
        "--source-layer",
        str(source_layer),
        "--read-layer",
        str(read_layer),
        "--alpha",
        str(alpha),
        "--max-train-prompts",
        str(max_train_prompts),
        "--max-eval-prompts",
        str(max_eval_prompts),
        "--seed",
        str(seed),
        "--out-dir",
        "steering_runs",
    ]
    if run_id:
        argv.extend(["--run-id", run_id])
    if sae_release and sae_id:
        argv.extend(["--sae-release", sae_release, "--sae-id", sae_id])
    if not prepend_bos:
        argv.append("--no-prepend-bos")
    if not steer_all_tokens:
        argv.append("--steer-last-token-only")

    print(run_steering_artifacts.remote(argv))
