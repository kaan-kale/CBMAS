"""Thin Modal wrapper for the Gemma alpha sweep experiment."""

from __future__ import annotations

import os
from pathlib import Path

import modal


APP_NAME = "cbmas-gemma-alpha-sweep"
REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = "/root"

graphs_vol = modal.Volume.from_name("cbmas-gemma-alpha-sweep", create_if_missing=True)
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
        "matplotlib>=3.9,<4",
        "torch>=2.6,<3",
        "transformers>=4.44,<5",
        "datasets>=2.20,<3",
        "sentencepiece>=0.2,<0.3",
        "transformer_lens>=1.15,<2",
        "jaxtyping>=0.2.33,<0.3",
        "typeguard>=2.13,<3",
        "pandas>=2.2,<3",
        "tqdm>=4.66,<5",
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
        f"{PROJECT_ROOT}/graphs/gemma_alpha_sweep": graphs_vol,
        f"{PROJECT_ROOT}/cache": cache_vol,
    },
    secrets=[hf_secret],
)
def run_gemma_alpha_sweep(argv: list[str]) -> str:
    """Run the local Gemma sweep module remotely."""
    os.chdir(PROJECT_ROOT)

    from experiments.gemma_alpha_sweep import main

    main(argv)
    graphs_vol.commit()
    cache_vol.commit()
    return "Finished Gemma alpha sweep"


@app.local_entrypoint()
def main(
    model_name: str = "google/gemma-2-2b-it",
    behavior_name: str = "reassurance",
    source_layer: int = 12,
    read_layer: int = 20,
    layer_pairs: str = "",
    source_site: str = "hook_resid_mid",
    read_site: str = "hook_resid_post",
    alpha_values: str = "-4,-2,-1,0,1,2,4",
    max_train_prompts: int = 40,
    max_eval_prompts: int = 30,
    num_prompts: int = 0,
    seed: int = 42,
    run_id: str = "",
    out_dir: str = "graphs/gemma_alpha_sweep",
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
        "--source-site",
        source_site,
        "--read-site",
        read_site,
        f"--alpha-values={alpha_values}",
        "--seed",
        str(seed),
        "--out-dir",
        out_dir,
    ]
    if layer_pairs:
        argv.extend(["--layer-pairs", layer_pairs])
    if num_prompts > 0:
        argv.extend(["--num-prompts", str(num_prompts)])
    else:
        argv.extend(
            [
                "--max-train-prompts",
                str(max_train_prompts),
                "--max-eval-prompts",
                str(max_eval_prompts),
            ]
        )
    if run_id:
        argv.extend(["--run-id", run_id])
    if not prepend_bos:
        argv.append("--no-prepend-bos")
    if not steer_all_tokens:
        argv.append("--steer-last-token-only")

    print(run_gemma_alpha_sweep.remote(argv))
