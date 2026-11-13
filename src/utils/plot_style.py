"""Shared matplotlib configuration for figures suitable for an IEEE-style paper.

Conventions follow common IEEE Transactions figure guidance: serif fonts, a
single-column width of 3.5 in (double-column 7.16 in), >=300 DPI raster export,
embedded (editable) fonts in vector output, muted dashed gridlines, and a
colorblind-safe categorical palette (Okabe & Ito, 2008) so lines stay
distinguishable in grayscale print.

All plotting code in this project should call `apply_ieee_style()` once before
drawing, then use `figsize(...)` / `new_figure(...)` / `save_figure(...)` so
every figure in the project (training curves and evaluation diagnostics alike)
looks like it belongs to the same paper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt

IEEE_COLUMN_WIDTH_IN = 3.5
IEEE_PAGE_WIDTH_IN = 7.16
IEEE_DPI = 300

# Okabe-Ito colorblind-safe palette, widely used in IEEE Vis/TVCG figures.
IEEE_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

_STYLE_APPLIED = False


def apply_ieee_style() -> None:
    """Configure matplotlib rcParams for IEEE-style figures. Idempotent."""
    global _STYLE_APPLIED
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.5,
        "axes.prop_cycle": matplotlib.cycler(color=IEEE_COLORS),
        "figure.dpi": 120,
        "savefig.dpi": IEEE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    _STYLE_APPLIED = True


def figsize(width: str = "single", aspect: float = 0.72) -> tuple:
    """Figure size in inches for a 'single'- or 'double'-column IEEE figure."""
    w = IEEE_COLUMN_WIDTH_IN if width == "single" else IEEE_PAGE_WIDTH_IN
    return (w, w * aspect)


def new_figure(width: str = "single", aspect: float = 0.72, **subplots_kwargs):
    """Create a styled (fig, ax) pair. Extra kwargs go to plt.subplots (e.g. nrows/ncols)."""
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=figsize(width, aspect), **subplots_kwargs)
    return fig, ax


def style_axis(ax, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
               title: Optional[str] = None, legend: bool = True, legend_loc: str = "best") -> None:
    """Apply axis labels, title, legend, and gridlines in the IEEE style."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc)
    ax.grid(True)


def save_figure(fig, path, also_pdf: bool = False) -> None:
    """Save at IEEE-ready resolution and close the figure to free memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches="tight")
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
