import json
from pathlib import Path

import pandas as pd

from dashboard.sae_data import (
    load_prompt_feature_details,
    load_prompt_map,
    neuronpedia_feature_url,
)


def test_neuronpedia_feature_url_uses_residual_gemma_scope():
    assert neuronpedia_feature_url(12, 2291) == (
        "https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/2291"
    )


def test_loads_prompt_text_and_feature_details(tmp_path):
    group = tmp_path / "source_8_read_12" / "alpha_10"
    summaries = group / "summaries"
    summaries.mkdir(parents=True)
    (group / "metadata.json").write_text(
        json.dumps(
            {
                "source_layer": 8,
                "read_layer": 12,
                "alpha": 10.0,
                "prompts": [{"prompt_id": 3, "prompt": "Will things improve?"}],
            }
        ),
        encoding="utf-8",
    )
    (summaries / "sae_feature_summary.json").write_text(
        json.dumps(
            [
                {
                    "prompt_id": 3,
                    "top_changed_features": [
                        {"feature": 42, "abs_mean_delta": 0.75, "mean_delta": -0.75}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    assert load_prompt_map(tmp_path) == {3: "Will things improve?"}
    result = load_prompt_feature_details(tmp_path)
    expected = pd.DataFrame(
        [
            {
                "source_layer": 8,
                "read_layer": 12,
                "alpha": 10.0,
                "prompt_id": 3,
                "prompt": "Will things improve?",
                "feature_id": 42,
                "rank": 1,
                "abs_mean_delta": 0.75,
                "mean_delta": -0.75,
                "direction": "decrease",
            }
        ]
    )
    pd.testing.assert_frame_equal(result, expected)


def test_local_artifact_directories_are_ignored():
    gitignore = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "/steering_runs/" in gitignore
    assert "/steering_run_*/" in gitignore
    assert "/20??-??-??_??-??-??_*/" in gitignore
