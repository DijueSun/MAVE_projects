from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTFILE = Path("Streamlit_app/sampling_variability_example.png")


def beta_params_from_mean_kappa(mean: float, kappa: float) -> tuple[float, float]:
    mean = float(np.clip(mean, 1e-6, 1 - 1e-6))
    kappa = float(max(kappa, 2.0))
    return mean * kappa, (1 - mean) * kappa


def main() -> None:
    rng = np.random.default_rng(42)

    mapping_mean = 0.40
    mapping_kappa = 60.0
    a, b = beta_params_from_mean_kappa(mapping_mean, mapping_kappa)
    mapping_samples = rng.beta(a, b, size=4000)

    library_size = 80
    skew_sigma = 0.50
    raw_weights = rng.lognormal(mean=0.0, sigma=skew_sigma, size=library_size)
    weights = raw_weights / raw_weights.sum()
    weights_sorted = np.sort(weights)[::-1]

    cells_transfected = 50_000
    cell_alloc = rng.multinomial(cells_transfected, weights)
    order = np.argsort(cell_alloc)[::-1]
    top_n = 20
    top_idx = order[:top_n]
    top_alloc = cell_alloc[top_idx]

    hdr_rate = 0.30
    edited_counts = rng.binomial(cell_alloc, hdr_rate)
    top_edited = edited_counts[top_idx]

    fig = plt.figure(figsize=(15, 10), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.hist(mapping_samples, bins=36, color="#4C78A8", edgecolor="white")
    ax1.axvline(mapping_mean, color="black", linestyle="--", lw=1.5)
    ax1.set_title("Beta: Mapping-Rate Variability", fontsize=15)
    ax1.set_xlabel("Mapping rate")
    ax1.set_ylabel("Sample count")
    ax1.text(
        0.04,
        0.96,
        r"$r_{map} \sim \mathrm{Beta}(\alpha,\beta)$" "\n"
        rf"$\alpha={a:.0f},\ \beta={b:.0f},\ \mu={mapping_mean:.2f}$",
        transform=ax1.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
    )

    ax2.plot(np.arange(1, library_size + 1), weights_sorted, color="#F58518", lw=2.5)
    ax2.fill_between(np.arange(1, library_size + 1), weights_sorted, color="#F58518", alpha=0.18)
    ax2.set_title("Lognormal: Library Skew", fontsize=15)
    ax2.set_xlabel("Variant rank")
    ax2.set_ylabel("Relative abundance")
    ax2.text(
        0.04,
        0.96,
        r"$w_i \sim \mathrm{LogNormal}(0,\sigma^2)$" "\n"
        rf"$\sigma={skew_sigma:.2f}$, then normalize to sum to 1",
        transform=ax2.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
    )

    ax3.bar(np.arange(1, top_n + 1), top_alloc, color="#54A24B", width=0.8)
    ax3.set_title("Multinomial: Cell Allocation", fontsize=15)
    ax3.set_xlabel("Top-ranked variants")
    ax3.set_ylabel("Allocated cells")
    ax3.text(
        0.04,
        0.96,
        r"$\mathbf{n} \sim \mathrm{Multinomial}(N_{cells},\mathbf{p})$" "\n"
        rf"$N_{{cells}}={cells_transfected:,}$",
        transform=ax3.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
    )

    x = np.arange(1, top_n + 1)
    ax4.bar(x, top_alloc, color="#C9D3E0", width=0.8, label="Allocated")
    ax4.bar(x, top_edited, color="#E45756", width=0.8, label="Edited")
    ax4.set_title("Binomial: Editing per Variant", fontsize=15)
    ax4.set_xlabel("Top-ranked variants")
    ax4.set_ylabel("Cell count")
    ax4.legend(frameon=False, loc="upper right")
    ax4.text(
        0.04,
        0.96,
        r"$e_i \sim \mathrm{Binomial}(n_i,\hat{h})$" "\n"
        rf"$\hat{{h}}={hdr_rate:.2f}$",
        transform=ax4.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
    )

    fig.suptitle(
        "Experimental Variability and Sampling in the Model",
        fontsize=19,
        y=0.965,
        weight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "These steps create experimental variability before final read counts and downstream QC metrics such as dropout and P10 are calculated.",
        ha="center",
        fontsize=11,
        style="italic",
    )

    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.07, right=0.98)
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTFILE}")


if __name__ == "__main__":
    main()
