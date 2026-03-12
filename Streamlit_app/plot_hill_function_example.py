from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTFILE = Path("Streamlit_app/hill_function_example.png")


def hill(x: np.ndarray, k: float, n: float = 2.0) -> np.ndarray:
    x = np.maximum(x.astype(float), 0.0)
    k = max(float(k), 1e-12)
    return (x ** n) / (k ** n + x ** n)


def main() -> None:
    hdr_x = np.linspace(0, 2000, 500)
    sgrna_x = np.linspace(0, 1500, 500)

    k_hdr = 500.0
    k_sg = 300.0
    hill_n = 2.0

    hdr_y = hill(hdr_x, k_hdr, hill_n)
    sgrna_y = hill(sgrna_x, k_sg, hill_n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    axes[0].plot(hdr_x, hdr_y, color="#1f77b4", lw=2.5)
    axes[0].axvline(k_hdr, color="#1f77b4", ls="--", lw=1.4, alpha=0.8)
    axes[0].axhline(0.5, color="#666666", ls=":", lw=1.2)
    axes[0].set_title("HDR Donor Hill Function")
    axes[0].set_xlabel("HDR_ng")
    axes[0].set_ylabel("Hill response")
    axes[0].set_xlim(0, 2000)
    axes[0].set_ylim(0, 1.05)
    axes[0].text(
        0.04,
        0.96,
        "hill(HDR_ng, K_hdr=500, n=2)\nK_hdr gives 50% response",
        transform=axes[0].transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
    )

    axes[1].plot(sgrna_x, sgrna_y, color="#d62728", lw=2.5)
    axes[1].axvline(k_sg, color="#d62728", ls="--", lw=1.4, alpha=0.8)
    axes[1].axhline(0.5, color="#666666", ls=":", lw=1.2)
    axes[1].set_title("sgRNA Hill Function")
    axes[1].set_xlabel("sgRNA_ng")
    axes[1].set_ylabel("Hill response")
    axes[1].set_xlim(0, 1500)
    axes[1].set_ylim(0, 1.05)
    axes[1].text(
        0.04,
        0.96,
        "hill(sgRNA_ng, K_sg=300, n=2)\nK_sg gives 50% response",
        transform=axes[1].transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
    )

    fig.suptitle(
        "Hill Functions Used in the HDR Model\nresponse = x^n / (K^n + x^n), with n = 2",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout()
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTFILE}")


if __name__ == "__main__":
    main()
