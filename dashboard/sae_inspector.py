"""Streamlit panel for exploring SAE changes caused by activation steering."""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.sae_data import (  # noqa: E402
    load_analysis_tables,
    load_prompt_feature_details,
    load_prompt_map,
    load_run_metadata,
    neuronpedia_feature_url,
    validate_run_dir,
)


DEFAULT_RUN_DIR = Path(
    os.environ.get(
        "CBMAS_SAE_RUN_DIR",
        PROJECT_ROOT / "2026-06-22_07-29-08_789c",
    )
)


st.set_page_config(page_title="CBMAS SAE Inspector", page_icon="🔬", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f4f0e8; color: #17251f; }
    [data-testid="stSidebar"] { background: #17251f; }
    [data-testid="stSidebar"] * { color: #f4f0e8; }
    div[data-testid="stMetric"] {
        background: #fffaf0; border: 1px solid #d9cfbd; border-radius: 14px; padding: 14px;
    }
    .feature-card {
        background: #fffaf0; border-left: 5px solid #d8683a; border-radius: 12px;
        padding: 16px 18px; margin: 8px 0 18px;
    }
    .prompt-box {
        background: #fffaf0; border: 1px solid #d9cfbd; border-radius: 12px;
        padding: 14px 16px; white-space: pre-wrap;
    }
    h1, h2, h3 { color: #173f35; font-family: Georgia, serif; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dashboard_data(run_dir_text: str):
    run_dir = validate_run_dir(Path(run_dir_text))
    tables = load_analysis_tables(run_dir)
    return (
        run_dir,
        tables,
        load_prompt_feature_details(run_dir),
        load_prompt_map(run_dir),
        load_run_metadata(run_dir),
    )


st.title("CBMAS SAE Feature Inspector")
st.caption("Dense steering sonrası gerçek read-layer activations içindeki SAE feature değişimlerini inceler.")

with st.sidebar:
    st.header("Run selection")
    run_dir_text = st.text_input("Local run directory", value=str(DEFAULT_RUN_DIR))
    st.caption("Panel Modal'a bağlanmaz; indirilmiş artifact klasörünü read-only olarak okur.")

try:
    run_dir, tables, details, prompt_map, metadata = load_dashboard_data(run_dir_text)
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(str(exc))
    st.info("Önce run klasörünü indirin ve analysis CSV'lerini üretin.")
    st.stop()

if details.empty:
    st.warning("Bu run içinde per-prompt SAE feature detail bulunamadı.")
    st.stop()

layers = sorted(int(value) for value in details["read_layer"].unique())
with st.sidebar:
    st.header("Filters")
    read_layer = st.selectbox("Read layer", layers, index=0)
    layer_details = details[details["read_layer"] == read_layer]
    alphas = sorted(float(value) for value in layer_details["alpha"].unique())
    default_alpha_index = alphas.index(max(alphas))
    alpha = st.select_slider("Steering alpha", options=alphas, value=alphas[default_alpha_index])
    prompt_options = ["All prompts"] + [f"Prompt {pid}" for pid in sorted(prompt_map)]
    prompt_choice = st.selectbox("Prompt", prompt_options)
    top_n = st.slider("Features shown", min_value=5, max_value=50, value=15, step=5)

filtered = layer_details[layer_details["alpha"] == alpha].copy()
if prompt_choice != "All prompts":
    selected_prompt_id = int(prompt_choice.removeprefix("Prompt "))
    filtered = filtered[filtered["prompt_id"] == selected_prompt_id]

feature_ranking = (
    filtered.groupby("feature_id", as_index=False)
    .agg(
        mean_abs_delta=("abs_mean_delta", "mean"),
        mean_delta=("mean_delta", "mean"),
        prompt_count=("prompt_id", "nunique"),
        hit_count=("feature_id", "size"),
    )
    .sort_values(["mean_abs_delta", "prompt_count"], ascending=[False, False])
)

summary_match = tables["comparison"][
    (tables["comparison"]["read_layer"] == read_layer)
    & (tables["comparison"]["alpha"] == alpha)
]
summary = summary_match.iloc[0] if not summary_match.empty else None

metric_cols = st.columns(4)
metric_cols[0].metric("Read layer", read_layer)
metric_cols[1].metric("Alpha", f"{alpha:g}")
metric_cols[2].metric("Prompts", int(filtered["prompt_id"].nunique()))
metric_cols[3].metric(
    "Mean SAE delta norm",
    f"{float(summary['mean_sae_delta_norm']):.3f}" if summary is not None else "n/a",
)

st.subheader("Top changed features")
st.caption(
    "Ranking mean absolute delta ile yapılır. Positive mean delta activation artışı, negative ise azalış gösterir."
)
chart_data = feature_ranking.head(top_n).set_index("feature_id")[["mean_abs_delta"]]
st.bar_chart(chart_data, color="#d8683a")

display_ranking = feature_ranking.head(top_n).copy()
display_ranking["neuronpedia"] = display_ranking["feature_id"].map(
    lambda feature_id: neuronpedia_feature_url(read_layer, int(feature_id))
)
st.dataframe(
    display_ranking,
    width="stretch",
    hide_index=True,
    column_config={
        "feature_id": st.column_config.NumberColumn("Feature ID", format="%d"),
        "mean_abs_delta": st.column_config.NumberColumn("Mean |delta|", format="%.4f"),
        "mean_delta": st.column_config.NumberColumn("Mean delta", format="%.4f"),
        "prompt_count": "Prompt count",
        "hit_count": "Hits",
        "neuronpedia": st.column_config.LinkColumn("Inspect on Neuronpedia", display_text="Open feature"),
    },
)

if feature_ranking.empty:
    st.info("Selected filters için feature bulunamadı.")
    st.stop()

feature_ids = [int(value) for value in feature_ranking["feature_id"].head(top_n)]
selected_feature = st.selectbox("Feature to inspect", feature_ids, format_func=lambda value: f"Feature {value}")
feature_url = neuronpedia_feature_url(read_layer, selected_feature)
selected_rows = filtered[filtered["feature_id"] == selected_feature].sort_values(
    ["abs_mean_delta", "prompt_id"], ascending=[False, True]
)

st.markdown(
    f"""
    <div class="feature-card">
      <strong>Layer {read_layer} · Feature {selected_feature}</strong><br>
      Bu tablodaki değerler kendi reassurance prompt'larımızdan geliyor. Semantic explanation ve
      public activation examples için <a href="{feature_url}" target="_blank" rel="noopener noreferrer">Neuronpedia feature sayfasını aç</a>.
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Prompt-level evidence")
prompt_table = selected_rows[
    ["prompt_id", "rank", "abs_mean_delta", "mean_delta", "direction"]
].copy()
st.dataframe(
    prompt_table,
    width="stretch",
    hide_index=True,
    column_config={
        "prompt_id": "Prompt ID",
        "rank": "Top-change rank",
        "abs_mean_delta": st.column_config.NumberColumn("|delta|", format="%.4f"),
        "mean_delta": st.column_config.NumberColumn("Signed delta", format="%.4f"),
        "direction": "Direction",
    },
)

for row in selected_rows.itertuples(index=False):
    label = f"Prompt {row.prompt_id} · {row.direction} · delta {row.mean_delta:+.4f}"
    with st.expander(label):
        safe_prompt = html.escape(str(row.prompt))
        st.markdown(f'<div class="prompt-box">{safe_prompt}</div>', unsafe_allow_html=True)

with st.expander("Run metadata"):
    st.json(
        {
            "run_directory": str(run_dir),
            "run_id": metadata.get("run_id"),
            "model_name": metadata.get("model_name"),
            "behavior_name": metadata.get("behavior_name"),
            "source_layer": metadata.get("source_layer"),
            "sae": metadata.get("sae"),
            "used_saelens": metadata.get("used_saelens"),
        }
    )
