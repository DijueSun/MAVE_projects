from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "Simulation_Prediction_modelling" / "26_feb_modelling"
TRAINING_CSV = MODEL_DIR / "synthetic_training_data_constraints.csv"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import sge_model_skew_dna_mapping_v4 as mod  # noqa: E402

LIBRARY_SIZE_MIN = 390
LIBRARY_SIZE_MAX = 2500
FIGURE_PATHS = {
    "workflow_a": APP_DIR / "workflow_A_surrogate_build.png",
    "workflow_b": APP_DIR / "workflow_B_slider_recommendation.png",
    "hill": APP_DIR / "hill_function_example.png",
    "hdr": APP_DIR / "hdr_rate_components.png",
    "precise_hdr": APP_DIR / "precise_hdr_event_equations.png",
    "monte_carlo": APP_DIR / "monte_carlo_model_example.png",
    "parameter_flow": APP_DIR / "parameter_flow_diagram.png",
    "sampling": APP_DIR / "sampling_variability_example.png",
}


@st.cache_data(show_spinner=False)
def load_training_rows() -> list[dict]:
    if TRAINING_CSV.exists():
        df = pd.read_csv(TRAINING_CSV)
        lib_min = int(df["library_size"].min())
        lib_max = int(df["library_size"].max())
        if lib_min <= LIBRARY_SIZE_MIN and lib_max >= LIBRARY_SIZE_MAX:
            return df.to_dict(orient="records")

    rng = np.random.default_rng(42)
    rows = mod.build_synthetic_dataset(
        n_samples=2500,
        n_reps_per_sample=10,
        hdr_range=(100.0, 2000.0),
        sgrna_range=(50.0, 1500.0),
        skew_range=(0.1, 1.5),
        mapping_range=(0.30, 0.70),
        reads_total=3_000_000,
        cells_transfected=5_600_000,
        library_size=2_000,
        library_size_range=(LIBRARY_SIZE_MIN, LIBRARY_SIZE_MAX),
        ratio_opt=2.0,
        rng=rng,
    )
    return rows


@st.cache_resource(show_spinner=False)
def fit_surrogates():
    rows = load_training_rows()
    return mod.fit_surrogate_models(rows)


def show_figure(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)
    else:
        st.warning(f"Missing figure: {path.name}")


@st.cache_data(show_spinner=False)
def run_search(
    hdr_ng: float,
    sgrna_ng: float,
    skew_sigma: float,
    mapping_target: float,
    mapping_window: float,
    target_dropout: float,
    min_p10_reads: float,
    reads_total: int,
    cells_transfected: int,
    library_low: int,
    library_high: int,
    n_candidates: int,
    top_k_verify: int,
    n_reps_verify: int,
    seed: int,
) -> pd.DataFrame:
    dropout_model, p10_model, feature_names = fit_surrogates()

    sgrna_range = (float(sgrna_ng), float(sgrna_ng))
    skew_range = (float(skew_sigma), float(skew_sigma))

    mapping_low = float(np.clip(mapping_target - mapping_window, 0.0, 1.0))
    mapping_high = float(np.clip(mapping_target + mapping_window, 0.0, 1.0))
    if mapping_high < mapping_low:
        mapping_high = mapping_low

    if library_high < library_low:
        library_high = library_low

    rng = np.random.default_rng(int(seed))
    df_verified = mod.suggest_experiments_surrogate_verified(
        dropout_model=dropout_model,
        p10_model=p10_model,
        feature_names=feature_names,
        n_candidates=int(n_candidates),
        top_k_to_verify=int(top_k_verify),
        target_dropout=float(target_dropout),
        min_p10_reads=float(min_p10_reads),
        min_mapping_rate=float(mapping_low),
        min_reads_total=int(reads_total),
        hdr_range=(float(hdr_ng), float(hdr_ng)),
        sgrna_range=sgrna_range,
        skew_range=skew_range,
        mapping_range=(float(mapping_low), float(mapping_high)),
        reads_total=int(reads_total),
        cells_transfected=int(cells_transfected),
        library_size=int((int(library_low) + int(library_high)) // 2),
        library_size_range=(int(library_low), int(library_high)),
        ratio_opt=2.0,
        n_reps_verify=int(n_reps_verify),
        rng=rng,
    )
    return df_verified


st.set_page_config(page_title="Library Mapping Slider Planner", layout="wide")
st.title("Library Mapping Slider Planner")
st.caption(
    "Move one experimental parameter (for example HDR ng), and this tool updates the other recommended "
    "parameters to target your desired library mapping."
)

with st.sidebar:
    st.subheader("Targets")
    target_mapping_pct = st.slider("Target library mapping (%)", min_value=30, max_value=70, value=40, step=1)
    mapping_window_pct = st.slider(
        "Mapping tolerance window (± %)", min_value=0, max_value=15, value=3, step=1
    )
    target_dropout_pct = st.slider("Max dropout target (%)", min_value=1, max_value=20, value=2, step=1)
    min_p10_reads = st.slider("Min P10 reads", min_value=10, max_value=500, value=100, step=5)

    st.subheader("Experiment Inputs")
    hdr_ng = st.slider("HDR ng", min_value=100, max_value=2000, value=700, step=10)
    sgrna_ng = st.slider("sgRNA ng", min_value=50, max_value=1500, value=350, step=10)
    skew_sigma = st.slider("Skew sigma", min_value=0.10, max_value=1.50, value=0.50, step=0.05)
    reads_total = st.slider("Reads total", min_value=1_000_000, max_value=10_000_000, value=3_000_000, step=100_000)
    cells_transfected = st.slider(
        "Effective transfected cells", min_value=1_000_000, max_value=12_000_000, value=5_600_000, step=100_000
    )
    library_low, library_high = st.slider(
        "Library size range",
        LIBRARY_SIZE_MIN,
        LIBRARY_SIZE_MAX,
        (LIBRARY_SIZE_MIN, LIBRARY_SIZE_MAX),
        step=10,
    )

    st.subheader("Precise HDR Event Calculator")
    transfection_eff_pct = st.slider("Transfection efficiency (%)", min_value=1, max_value=100, value=60, step=1)
    precise_hdr_given_hdr_pct = st.slider(
        "Precise HDR among HDR edits (%)", min_value=1, max_value=100, value=100, step=1
    )
    target_precise_hdr_events = st.slider(
        "Target precise HDR events", min_value=1_000, max_value=5_000_000, value=100_000, step=1_000
    )

    with st.expander("Advanced Search Settings"):
        st.caption("These only affect how broadly the app searches. They are not biological inputs.")
        n_candidates = st.slider("Candidate samples", min_value=2000, max_value=30000, value=10000, step=1000)
        top_k_verify = st.slider("Top candidates to verify", min_value=10, max_value=100, value=40, step=5)
        n_reps_verify = st.slider("Monte Carlo reps per verification", min_value=10, max_value=80, value=30, step=5)
        seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)

mapping_target = float(target_mapping_pct) / 100.0
mapping_window = float(mapping_window_pct) / 100.0
target_dropout = float(target_dropout_pct) / 100.0

with st.spinner("Searching recommendations..."):
    df_verified = run_search(
        hdr_ng=float(hdr_ng),
        sgrna_ng=float(sgrna_ng),
        skew_sigma=float(skew_sigma),
        mapping_target=float(mapping_target),
        mapping_window=float(mapping_window),
        target_dropout=float(target_dropout),
        min_p10_reads=float(min_p10_reads),
        reads_total=int(reads_total),
        cells_transfected=int(cells_transfected),
        library_low=int(library_low),
        library_high=int(library_high),
        n_candidates=int(n_candidates),
        top_k_verify=int(top_k_verify),
        n_reps_verify=int(n_reps_verify),
        seed=int(seed),
    )

if df_verified.empty:
    st.error("No feasible candidates were returned. Widen the ranges or increase candidate samples.")
    st.stop()

best = df_verified.iloc[0]
transfection_eff = float(transfection_eff_pct) / 100.0
precise_hdr_given_hdr = float(precise_hdr_given_hdr_pct) / 100.0

precise_hdr_fraction_tx = float(best["hdr_rate_pred"]) * precise_hdr_given_hdr
precise_hdr_fraction_pop = mod.precise_hdr_fraction_population(
    hdr_rate=float(best["hdr_rate_pred"]),
    transfection_rate=transfection_eff,
    precise_hdr_given_hdr=precise_hdr_given_hdr,
)
pred_precise_events_tx = mod.precise_hdr_events_haploid(
    n_cells=int(best["cells_transfected"]),
    precise_hdr_fraction=precise_hdr_fraction_tx,
)
required_effective_tx_cells = mod.required_cells_for_precise_hdr_events(
    target_precise_hdr_events=float(target_precise_hdr_events),
    precise_hdr_fraction=precise_hdr_fraction_tx,
)
required_total_cells_at_tx = mod.required_cells_for_precise_hdr_events(
    target_precise_hdr_events=float(target_precise_hdr_events),
    precise_hdr_fraction=precise_hdr_fraction_pop,
)
required_effective_tx_cells_display = (
    f"{required_effective_tx_cells:,.0f}" if np.isfinite(required_effective_tx_cells) else "N/A"
)
required_total_cells_at_tx_display = (
    f"{required_total_cells_at_tx:,.0f}" if np.isfinite(required_total_cells_at_tx) else "N/A"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted HDR rate", f"{100.0 * float(best['hdr_rate_pred']):.1f}%")
col2.metric("Verified mapping", f"{100.0 * float(best['mapping_rate_mean']):.1f}%")
col3.metric(
    "Verified dropout",
    f"{100.0 * float(best['dropout_sim_mean']):.2f}%",
    help="Fraction of library variants with zero reads after Monte Carlo verification. Lower is better.",
)
col4.metric(
    "Verified P10 reads",
    f"{float(best['p10_reads_sim_mean']):.1f}",
    help="The 10th percentile of read depth across variants after Monte Carlo verification. Higher means better low-end coverage.",
)

col5, col6, col7, col8 = st.columns(4)
col5.metric("Pred precise HDR in tx cells", f"{100.0 * precise_hdr_fraction_tx:.1f}%")
col6.metric("Pred precise HDR in population", f"{100.0 * precise_hdr_fraction_pop:.1f}%")
col7.metric("Pred precise HDR events", f"{pred_precise_events_tx:,.0f}")
col8.metric("Cells needed for target events", required_total_cells_at_tx_display)

st.subheader("Recommended Experiment Settings")
st.write(
    pd.DataFrame(
        [
            {
                "HDR_ng": round(float(best["HDR_ng"]), 2),
                "sgRNA_ng": round(float(best["sgRNA_ng"]), 2),
                "skew_sigma": round(float(best["skew_sigma"]), 3),
                "mapping_rate_assumed_%": round(100.0 * float(best["mapping_rate_assumed"]), 2),
                "reads_total": int(best["reads_total"]),
                "cells_transfected": int(best["cells_transfected"]),
                "library_size": int(best["library_size"]),
                "transfection_eff_%": round(100.0 * transfection_eff, 2),
                "precise_hdr_given_hdr_%": round(100.0 * precise_hdr_given_hdr, 2),
                "precise_hdr_fraction_tx_%": round(100.0 * precise_hdr_fraction_tx, 2),
                "precise_hdr_fraction_population_%": round(100.0 * precise_hdr_fraction_pop, 2),
                "pred_precise_hdr_events": round(pred_precise_events_tx, 0),
                "required_effective_tx_cells": required_effective_tx_cells_display,
                "required_total_cells_at_tx": required_total_cells_at_tx_display,
            }
        ]
    )
)

st.caption(
    "Experiment sliders rerun the model immediately. The advanced search settings only control how many "
    "candidate designs are sampled and verified."
)
st.caption(
    "Assumption used for the event calculator: predicted HDR rate is the HDR fraction among effectively "
    "transfected cells, and precise HDR fraction = HDR rate x precise HDR among HDR edits."
)

st.subheader("Top Verified Candidates")
st.caption(
    "These are the best candidate experiment settings after fast surrogate screening and Monte Carlo verification, "
    "ranked by lowest verified dropout and then highest verified P10 reads."
)
display_cols = [
    "HDR_ng",
    "sgRNA_ng",
    "skew_sigma",
    "hdr_rate_pred",
    "mapping_rate_assumed",
    "reads_total",
    "cells_transfected",
    "library_size",
    "dropout_sim_mean",
    "p10_reads_sim_mean",
    "mapping_rate_mean",
    "reads_usable_mean",
]
table = df_verified[display_cols].copy()
table["precise_hdr_fraction_tx_%"] = 100.0 * table["hdr_rate_pred"] * precise_hdr_given_hdr
table["precise_hdr_fraction_population_%"] = 100.0 * table["hdr_rate_pred"] * precise_hdr_given_hdr * transfection_eff
table["pred_precise_hdr_events"] = table["hdr_rate_pred"] * precise_hdr_given_hdr * table["cells_transfected"]
table["mapping_rate_assumed"] = 100.0 * table["mapping_rate_assumed"]
table["mapping_rate_mean"] = 100.0 * table["mapping_rate_mean"]
table["dropout_sim_mean"] = 100.0 * table["dropout_sim_mean"]
table.rename(
    columns={
        "hdr_rate_pred": "hdr_rate_pred_%",
        "mapping_rate_assumed": "mapping_rate_assumed_%",
        "mapping_rate_mean": "mapping_rate_mean_%",
        "dropout_sim_mean": "dropout_sim_mean_%",
    },
    inplace=True,
)
table["hdr_rate_pred_%"] = 100.0 * table["hdr_rate_pred_%"]
st.dataframe(
    table[
        [
            "HDR_ng",
            "sgRNA_ng",
            "skew_sigma",
            "hdr_rate_pred_%",
            "mapping_rate_assumed_%",
            "reads_total",
            "cells_transfected",
            "library_size",
            "dropout_sim_mean_%",
            "p10_reads_sim_mean",
            "mapping_rate_mean_%",
            "reads_usable_mean",
            "precise_hdr_fraction_tx_%",
            "precise_hdr_fraction_population_%",
            "pred_precise_hdr_events",
        ]
    ].head(20),
    use_container_width=True,
)

with st.expander("How The Model Works"):
    tab_workflow, tab_hdr, tab_sampling, tab_events = st.tabs(
        ["Workflow", "HDR Model", "Sampling", "Precise HDR"]
    )

    with tab_workflow:
        col_a, col_b = st.columns(2)
        with col_a:
            show_figure(FIGURE_PATHS["workflow_a"], "Workflow A: build the surrogate model")
        with col_b:
            show_figure(FIGURE_PATHS["workflow_b"], "Workflow B: use the slider tool to rank experiments")

    with tab_hdr:
        col_a, col_b = st.columns(2)
        with col_a:
            show_figure(FIGURE_PATHS["hill"], "Hill functions used for HDR donor and sgRNA dose-response")
        with col_b:
            show_figure(FIGURE_PATHS["hdr"], "How the final HDR rate is built from Hill and ratio terms")

    with tab_sampling:
        col_a, col_b = st.columns(2)
        with col_a:
            show_figure(FIGURE_PATHS["parameter_flow"], "Which parameters feed the HDR equation, simulator, and outputs")
            show_figure(FIGURE_PATHS["monte_carlo"], "Monte Carlo verification for one example candidate")
        with col_b:
            show_figure(FIGURE_PATHS["sampling"], "Beta, lognormal, multinomial, and binomial variability examples")

    with tab_events:
        show_figure(FIGURE_PATHS["precise_hdr"], "Precise HDR event equations used by the calculator")
