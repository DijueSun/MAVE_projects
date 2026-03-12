from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTFILE = Path("Streamlit_app/precise_hdr_event_equations.png")


def add_box(ax, x: float, y: float, w: float, h: float, text: str, fc: str, fs: int = 14) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.02",
        fc=fc,
        ec="#222222",
        lw=1.3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        "Precise HDR Event Equations Used in the Slider Tool",
        ha="center",
        va="top",
        fontsize=20,
        weight="bold",
        transform=ax.transAxes,
    )

    add_box(
        ax,
        0.06,
        0.67,
        0.40,
        0.16,
        "Precise HDR fraction\nin transfected cells\n\n"
        r"$f_{prec,tx} = \hat{h} \cdot f_{prec\mid HDR}$",
        "#DDEBF7",
    )
    add_box(
        ax,
        0.54,
        0.67,
        0.40,
        0.16,
        "Precise HDR fraction\nin the full population\n\n"
        r"$f_{prec,pop} = \hat{h} \cdot f_{tx} \cdot f_{prec\mid HDR}$",
        "#E2F0D9",
    )
    add_box(
        ax,
        0.06,
        0.40,
        0.40,
        0.16,
        "Expected precise HDR events\nin haploid HAP1\n\n"
        r"$N_{prec} = N_{tx} \cdot f_{prec,tx}$",
        "#FCE4D6",
    )
    add_box(
        ax,
        0.54,
        0.40,
        0.40,
        0.16,
        "Total cells needed\nfor a target event count\n\n"
        r"$N_{cells} = \dfrac{N_{target}}{f_{prec,pop}}$",
        "#E4DFEC",
    )
    add_box(
        ax,
        0.18,
        0.12,
        0.64,
        0.19,
        "Worked example\n\n"
        r"$\hat{h} = 0.30,\quad f_{tx} = 0.60,\quad f_{prec\mid HDR} = 0.80$"
        "\n"
        r"$f_{prec,pop} = 0.30 \times 0.60 \times 0.80 = 0.144$"
        "\n"
        r"$N_{cells} = \dfrac{100{,}000}{0.144} \approx 694{,}444$",
        "#FFF2CC",
        fs=13,
    )

    ax.text(
        0.5,
        0.04,
        r"Notation: $\hat{h}$ = predicted HDR rate, $f_{tx}$ = transfection efficiency, $f_{prec\mid HDR}$ = precise HDR among HDR edits.",
        ha="center",
        va="bottom",
        fontsize=12,
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
