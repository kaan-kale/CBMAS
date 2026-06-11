"""Thin Modal wrapper for the existing CBMAS CLI.

This file intentionally keeps Modal-specific logic separate from the core
experiment code. It only packages the repository, mounts data/cache volumes,
and forwards arguments into the existing CLI entrypoint.

Usage:
  pip install modal
  modal setup
  modal run modal_brc.py -- --help
  modal run modal_brc.py -- --dataset reassurance --metric logit_diffs --inject-layers 0 --read-layers 1
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


APP_NAME = "cbmas-brc"
REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = "/root"

graphs_vol = modal.Volume.from_name("cbmas-graphs", create_if_missing=True)
cache_vol = modal.Volume.from_name("cbmas-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.26,<2.0",
        "matplotlib>=3.9,<4",
        "torch>=2.6,<3",
        "transformers>=4.44,<5",
        "datasets>=2.20,<3",
        "transformer_lens>=1.15,<2",
        "jaxtyping>=0.2.33,<0.3",
        "typeguard>=2.13,<3",
        "pandas>=2.2,<3",
        "tqdm>=4.66,<5",
    )
    .add_local_python_source("BRC_Experiment")
    .add_local_dir(str(REPO_ROOT / "data"), remote_path=f"{PROJECT_ROOT}/data")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={
        f"{PROJECT_ROOT}/graphs_modal": graphs_vol,
        f"{PROJECT_ROOT}/cache": cache_vol,
    },
)
def run_cli(argv: list[str]) -> str:
    """Run the existing CLI inside Modal and persist generated outputs."""
    os.chdir(PROJECT_ROOT)

    from BRC_Experiment.Modularized.cli import main as cli_main

    cli_main(argv)
    graphs_vol.commit()
    cache_vol.commit()
    return "Finished experiment"


@app.local_entrypoint()
def main(
    model_name: str = "gpt2-small",
    dataset: str = "reassurance",
    metric: str = "logit_diffs",
    alpha_start: float = -1.0,
    alpha_stop: float = 1.0,
    alpha_step: float = 1.0,
    inject_layers: str = "0",
    read_layers: str = "1",
    inject_site: str = "hook_resid_mid",
    read_site: str = "hook_resid_post",
    prepend_bos: bool = True,
    steer_all_tokens: bool = True,
    use_log_scale: bool = False,
    log_scale_both: bool = False,
    seed: int = 42,
    show_progress: bool = False,
):
    argv = [
        "--model-name",
        model_name,
        "--dataset",
        dataset,
        "--out-dir",
        "graphs_modal",
        "--alpha-start",
        str(alpha_start),
        "--alpha-stop",
        str(alpha_stop),
        "--alpha-step",
        str(alpha_step),
        "--inject-layers",
        inject_layers,
        "--read-layers",
        read_layers,
        "--inject-site",
        inject_site,
        "--read-site",
        read_site,
        "--seed",
        str(seed),
    ]
    if not prepend_bos:
        argv.append("--no-prepend-bos")
    if not steer_all_tokens:
        argv.append("--steer-last-token-only")
    if use_log_scale:
        argv.append("--use-log-scale")
    if log_scale_both:
        argv.append("--log-scale-both")
    if show_progress:
        argv.append("--show-progress")
    if metric != "all":
        argv.extend(["--metric", metric])

    result = run_cli.remote(argv)
    print(result)
