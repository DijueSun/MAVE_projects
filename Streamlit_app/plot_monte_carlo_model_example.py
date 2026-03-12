from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Simulation_Prediction_modelling" / "26_feb_modelling"
OUTFILE = Path("Streamlit_app/monte_carlo_model_example.png")

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import sge_model_skew_dna_mapping_v4 as mod  # noqa: E402


def main() -> None:
    params = {
        "hdr_ng": 700.0,
        "sgrna_ng": 350.0,
        "ratio_opt": 2.0,
        "skew_sigma": 0.5,
        "mapping_rate": 0.40,
        "reads_total": 3_000_000,
        "cells_transfected": 5_600_000,
        "library_size": 1500,
    }
    n_reps = 200
    rng = np.random.default_rng(42)
    results = [mod.simulate_once(rng=rng, **params) for _ in range(n_reps)]

    dropout = np.array([r["dropout_frac"] for r in results], dtype=float)
    p10 = np.array([r["p10_reads"] for r in results], dtype=float)
    usable = np.array([r["reads_usable"] for r in results], dtype=float)
    hdr_rate = np.array([r["hdr_rate"] for r in results], dtype=float)

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.7, 1.2], hspace=0.35, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])

    ax0.axis("off")
    ax0.text(
        0.5,
        0.95,
        "How Monte Carlo is Used in This Model",
        ha="center",
        va="top",
        fontsize=20,
        weight="bold",
        transform=ax0.transAxes,
    )
    ax0.text(
        0.5,
        0.56,
        "1) Choose one candidate experiment\n"
        "2) Run simulate_once many times with stochastic cell allocation, editing, and read sampling\n"
        "3) Summarize the distribution of outputs across replicates\n"
        "4) Use mean and quantiles to verify candidate quality",
        ha="center",
        va="center",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.5", fc="#F7F7F7", ec="#333333"),
        transform=ax0.transAxes,
    )
    ax0.text(
        0.5,
        0.14,
        "Example candidate: HDR=700 ng, sgRNA=350 ng, mapping=40%, reads=3M, effective tx cells=5.6M, library size=1500",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax0.transAxes,
    )

    ax1.hist(100 * dropout, bins=22, color="#4C78A8", edgecolor="white")
    ax1.axvline(100 * dropout.mean(), color="black", ls="--", lw=1.5)
    ax1.set_title("Dropout Distribution")
    ax1.set_xlabel("Dropout (%)")
    ax1.set_ylabel("Replicate count")
    ax1.text(
        0.98,
        0.95,
        f"mean={100*dropout.mean():.2f}%\nq05={100*np.quantile(dropout, 0.05):.2f}%\nq95={100*np.quantile(dropout, 0.95):.2f}%",
        ha="right",
        va="top",
        transform=ax1.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
    )

    ax2.hist(p10, bins=22, color="#F58518", edgecolor="white")
    ax2.axvline(p10.mean(), color="black", ls="--", lw=1.5)
    ax2.set_title("P10 Read Distribution")
    ax2.set_xlabel("P10 reads")
    ax2.set_ylabel("Replicate count")
    ax2.text(
        0.98,
        0.95,
        f"mean={p10.mean():.1f}\nq05={np.quantile(p10, 0.05):.1f}\nq95={np.quantile(p10, 0.95):.1f}",
        ha="right",
        va="top",
        transform=ax2.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
    )

    ax3.hist(usable / 1_000_000, bins=18, color="#54A24B", edgecolor="white")
    ax3.axvline(usable.mean() / 1_000_000, color="black", ls="--", lw=1.5)
    ax3.set_title("Usable Reads Distribution")
    ax3.set_xlabel("Usable reads (millions)")
    ax3.set_ylabel("Replicate count")
    ax3.text(
        0.98,
        0.95,
        f"hdr rate={100*hdr_rate.mean():.1f}%\nusable mean={usable.mean()/1_000_000:.2f}M\nmapping fixed={100*params['mapping_rate']:.0f}%",
        ha="right",
        va="top",
        transform=ax3.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
    )

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTFILE}")


if __name__ == "__main__":
    main()
