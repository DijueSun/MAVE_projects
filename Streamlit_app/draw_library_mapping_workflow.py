from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw_box(ax, x: float, y: float, w: float, h: float, text: str, color: str, fs: float = 14) -> None:
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        fc=color,
        ec="#111111",
        lw=1.2,
    )
    ax.add_patch(rect)
    wrapped = textwrap.fill(text, width=32)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        wrapped,
        ha="center",
        va="center",
        fontsize=fs,
        linespacing=1.1,
        wrap=True,
        clip_on=True,
    )


def draw_arrow(ax, p1: tuple[float, float], p2: tuple[float, float], lw: float = 1.6) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=lw,
            color="#111111",
            connectionstyle="arc3,rad=0.0",
        )
    )


def draw_vertical_flow(ax, x: float, upper_y: float, lower_y: float, box_h: float, lw: float = 1.6) -> None:
    gap = upper_y - (lower_y + box_h)
    margin = max(0.01, min(0.025, gap * 0.25))
    draw_arrow(ax, (x, upper_y - margin), (x, lower_y + box_h + margin), lw=lw)


def build_vertical_page(title: str, steps: list[tuple[str, str]], colors: list[str], footer: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 13.0))
    ax.axis("off")
    ax.text(0.5, 0.985, title, ha="center", va="top", fontsize=18, weight="bold", transform=ax.transAxes)

    x = 0.12
    w = 0.76
    h = 0.082
    gap = 0.075
    y = 0.82

    for i, ((step_id, sentence), color) in enumerate(zip(steps, colors)):
        draw_box(ax, x, y, w, h, f"{step_id}. {sentence}", color, fs=15)
        if i < len(steps) - 1:
            next_y = y - (h + gap)
            draw_vertical_flow(ax, x + w / 2, y, next_y, h, lw=1.8)
            y = next_y

    ax.text(0.5, 0.06, footer, ha="center", va="center", fontsize=11, style="italic", transform=ax.transAxes)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw split workflow figures for the Library Mapping Slider Tool.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("Streamlit_app"),
        help="Output directory (default: Streamlit_app)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300)")
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also export PDFs with the same filename stem.",
    )
    args = parser.parse_args()

    colors_a = ["#DDEBF7", "#E2F0D9", "#E2F0D9", "#D9EAD3"]
    steps_a = [
        ("1", "Set the HDR, sgRNA, mapping, and library-size ranges."),
        ("2", "Simulate synthetic designs with the HDR, mapping, and skew model."),
        ("3", "Average repeated runs to estimate HDR, dropout, and P10."),
        ("4", "Fit fast Ridge surrogates for dropout and P10."),
    ]
    fig_a = build_vertical_page(
        "Workflow A: Build the Surrogate Model",
        steps_a,
        colors_a,
        "Output: trained surrogate models for dropout and P10 screening.",
    )

    colors_b = ["#FFF2CC", "#FCE4D6", "#FCE4D6", "#E4DFEC", "#D9EAD3"]
    steps_b = [
        ("1", "Set the target and experiment inputs with sliders."),
        ("2", "Sample candidate designs across the allowed ranges."),
        ("3", "Filter candidates with surrogate dropout and P10 predictions."),
        ("4", "Verify the top candidates with Monte Carlo reruns."),
        ("5", "Rank verified designs and report the best settings."),
    ]
    fig_b = build_vertical_page(
        "Workflow B: Use the Slider Tool to Rank Experiments",
        steps_b,
        colors_b,
        "Output: a ranked list of feasible experimental settings for the requested mapping target.",
    )

    args.outdir.mkdir(parents=True, exist_ok=True)

    out_a = args.outdir / "workflow_A_surrogate_build.png"
    out_b = args.outdir / "workflow_B_slider_recommendation.png"
    fig_a.savefig(out_a, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig_b.savefig(out_b, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved workflow image: {out_a}")
    print(f"Saved workflow image: {out_b}")

    if args.pdf:
        pdf_a = out_a.with_suffix(".pdf")
        pdf_b = out_b.with_suffix(".pdf")
        fig_a.savefig(pdf_a, bbox_inches="tight", facecolor="white")
        fig_b.savefig(pdf_b, bbox_inches="tight", facecolor="white")
        print(f"Saved workflow PDF: {pdf_a}")
        print(f"Saved workflow PDF: {pdf_b}")

    plt.close(fig_a)
    plt.close(fig_b)


if __name__ == "__main__":
    main()
