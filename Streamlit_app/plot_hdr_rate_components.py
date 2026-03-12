from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTFILE = Path("Streamlit_app/hdr_rate_components.png")


def hill(x: np.ndarray, k: float, n: float = 2.0) -> np.ndarray:
    x = np.maximum(x.astype(float), 0.0)
    k = max(float(k), 1e-12)
    return (x ** n) / (k ** n + x ** n)


def ratio_penalty(ratio: np.ndarray, ratio_opt: float = 2.0, ratio_sigma: float = 1.0) -> np.ndarray:
    ratio_sigma = max(float(ratio_sigma), 1e-12)
    return np.exp(-((ratio - ratio_opt) ** 2) / (2.0 * ratio_sigma ** 2))


def dna_to_hdr_rate(
    hdr_ng: np.ndarray,
    sgrna_ng: np.ndarray,
    r_min: float = 0.01,
    r_max: float = 0.60,
    k_hdr: float = 500.0,
    k_sg: float = 300.0,
    ratio_opt: float = 2.0,
    ratio_sigma: float = 1.0,
    hill_n: float = 2.0,
) -> np.ndarray:
    hdr_component = hill(hdr_ng, k_hdr, hill_n)
    sgrna_component = hill(sgrna_ng, k_sg, hill_n)
    ratio = hdr_ng / np.maximum(sgrna_ng, 1e-12)
    ratio_term = ratio_penalty(ratio, ratio_opt=ratio_opt, ratio_sigma=ratio_sigma)
    rate = r_min + (r_max - r_min) * hdr_component * sgrna_component * ratio_term
    return np.clip(rate, 0.0, 1.0)


def main() -> None:
    k_hdr = 500.0
    k_sg = 300.0
    hill_n = 2.0
    ratio_opt = 2.0
    ratio_sigma = 1.0
    r_min = 0.01
    r_max = 0.60

    hdr_grid = np.linspace(0, 2000, 400)
    sgrna_grid = np.linspace(0, 1500, 400)
    ratio_grid = np.linspace(0.05, 6.0, 400)

    hdr_curve = hill(hdr_grid, k_hdr, hill_n)
    sgrna_curve = hill(sgrna_grid, k_sg, hill_n)
    ratio_curve = ratio_penalty(ratio_grid, ratio_opt, ratio_sigma)

    hdr_mesh, sgrna_mesh = np.meshgrid(np.linspace(0, 2000, 220), np.linspace(1, 1500, 220))
    rate_mesh = dna_to_hdr_rate(
        hdr_mesh,
        sgrna_mesh,
        r_min=r_min,
        r_max=r_max,
        k_hdr=k_hdr,
        k_sg=k_sg,
        ratio_opt=ratio_opt,
        ratio_sigma=ratio_sigma,
        hill_n=hill_n,
    )

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.28, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.plot(hdr_grid, hdr_curve, color="#1f77b4", lw=2.5, label="HDR Hill term")
    ax1.plot(sgrna_grid, sgrna_curve, color="#d62728", lw=2.5, label="sgRNA Hill term")
    ax1.axvline(k_hdr, color="#1f77b4", ls="--", lw=1.2, alpha=0.8)
    ax1.axvline(k_sg, color="#d62728", ls="--", lw=1.2, alpha=0.8)
    ax1.axhline(0.5, color="#666666", ls=":", lw=1.1)
    ax1.set_xlim(0, 2000)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("DNA amount (ng)")
    ax1.set_ylabel("Response")
    ax1.set_title("Hill Terms")
    ax1.legend(frameon=False, loc="lower right")
    ax1.text(
        0.03,
        0.97,
        r"$H(x)=\dfrac{x^n}{K^n+x^n}$",
        transform=ax1.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc"),
    )

    ax2.plot(ratio_grid, ratio_curve, color="#2ca02c", lw=2.5)
    ax2.axvline(ratio_opt, color="#2ca02c", ls="--", lw=1.2, alpha=0.8)
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel(r"$HDR_{ng} / sgRNA_{ng}$")
    ax2.set_ylabel("Ratio term")
    ax2.set_title("Gaussian Ratio Term")
    ax2.text(
        0.04,
        0.96,
        r"$G(r)=e^{-(r-r_{opt})^2/(2\sigma_r^2)}$"
        "\n"
        r"$r_{opt}=2.0,\ \sigma_r=1.0$",
        transform=ax2.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
    )

    im = ax3.imshow(
        rate_mesh,
        origin="lower",
        aspect="auto",
        extent=[0, 2000, 1, 1500],
        cmap="viridis",
        vmin=r_min,
        vmax=r_max,
    )
    ax3.set_xlabel(r"$HDR_{ng}$")
    ax3.set_ylabel(r"$sgRNA_{ng}$")
    ax3.set_title("Predicted HDR Rate Surface")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.025, pad=0.02)
    cbar.set_label("Predicted HDR rate")

    fig.suptitle(
        "Predicted HDR Rate Model\n"
        r"$\hat{h}=r_{min}+(r_{max}-r_{min})\cdot H(HDR)\cdot H(sgRNA)\cdot G(ratio)$",
        fontsize=16,
        y=0.98,
    )

    fig.text(
        0.5,
        0.02,
        rf"$r_{{min}}={r_min},\ r_{{max}}={r_max},\ K_{{HDR}}={k_hdr},\ K_{{sgRNA}}={k_sg},\ "
        rf"r_{{opt}}={ratio_opt},\ \sigma_r={ratio_sigma},\ n={hill_n}$",
        ha="center",
        fontsize=11,
    )

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTFILE}")


if __name__ == "__main__":
    main()
