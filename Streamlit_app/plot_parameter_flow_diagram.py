from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTFILE = Path("Streamlit_app/parameter_flow_diagram.png")


def add_box(ax, x: float, y: float, w: float, h: float, text: str, fc: str, fs: int = 13) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        fc=fc,
        ec="#222222",
        lw=1.3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def add_arrow(ax, p1: tuple[float, float], p2: tuple[float, float], lw: float = 1.5) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=15,
            lw=lw,
            color="#222222",
            connectionstyle="arc3,rad=0.0",
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "How Parameters Flow Through the Library Mapping Model",
        ha="center",
        va="top",
        fontsize=19,
        weight="bold",
        transform=ax.transAxes,
    )

    # Left column: inputs
    add_box(ax, 0.05, 0.66, 0.24, 0.17, "HDR_ng\nsgRNA_ng\nHDR:sgRNA ratio", "#DDEBF7", fs=15)
    add_box(ax, 0.05, 0.38, 0.24, 0.19, "Effective transfected cells\nMapping rate\nReads total\nLibrary size\nSkew sigma", "#FCE4D6", fs=15)
    add_box(ax, 0.05, 0.14, 0.24, 0.12, "Transfection efficiency\nPrecise HDR fraction", "#FFF2CC", fs=14)

    # Middle column: model blocks
    add_box(ax, 0.38, 0.66, 0.24, 0.17, "HDR equation\nHill(HDR)\nHill(sgRNA)\nGaussian ratio term", "#E2F0D9", fs=15)
    add_box(ax, 0.38, 0.38, 0.24, 0.19, "Stochastic simulator\nCell allocation\nEditing draws\nRead sampling", "#E4DFEC", fs=15)
    add_box(ax, 0.38, 0.14, 0.24, 0.12, "Event calculator\nExpected precise HDR\nRequired cell count", "#D9EAD3", fs=14)

    # Right column: outputs
    add_box(ax, 0.72, 0.62, 0.23, 0.21, "Predicted HDR rate\nDropout\nP10 reads\nUsable reads", "#F4F4F4", fs=15)
    add_box(ax, 0.72, 0.26, 0.23, 0.17, "Precise HDR fraction\nPrecise HDR events\nCells needed", "#F4F4F4", fs=15)

    # Arrows
    add_arrow(ax, (0.29, 0.745), (0.38, 0.745))
    add_arrow(ax, (0.29, 0.475), (0.38, 0.475))
    add_arrow(ax, (0.29, 0.20), (0.38, 0.20))

    add_arrow(ax, (0.62, 0.745), (0.72, 0.72))
    add_arrow(ax, (0.62, 0.475), (0.72, 0.69))
    add_arrow(ax, (0.62, 0.20), (0.72, 0.345))

    add_arrow(ax, (0.50, 0.66), (0.50, 0.57))
    add_arrow(ax, (0.50, 0.38), (0.50, 0.26))

    ax.text(0.5, 0.60, "HDR rate feeds into simulation", ha="center", va="center", fontsize=11, color="#444444")
    ax.text(0.5, 0.31, "Simulation plus assumptions feed event estimates", ha="center", va="center", fontsize=11, color="#444444")

    ax.text(
        0.5,
        0.05,
        "Key idea: only HDR_ng and sgRNA_ng enter the Hill-based HDR equation; the other experimental parameters affect stochastic coverage and downstream QC outputs.",
        ha="center",
        va="center",
        fontsize=11,
        style="italic",
        transform=ax.transAxes,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTFILE}")


if __name__ == "__main__":
    main()
