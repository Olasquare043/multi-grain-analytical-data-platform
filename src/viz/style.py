"""Shared figure styling.

These figures go into an academic paper, so the rules are fixed once here rather
than negotiated per chart (section 12):

* **Colour-blind safe.** The categorical palette is Okabe-Ito, designed to stay
  distinguishable under deuteranopia, protanopia and tritanopia. Sequential maps
  are perceptually uniform (viridis family), never the rainbow maps that
  manufacture false boundaries.
* **Never colour alone.** Line charts vary dash pattern and marker as well as
  hue, so a monochrome print of the paper remains readable.
* **Units on every axis, source on every figure.** A figure that leaves the page
  must carry its own provenance.
* **Deterministic.** A fixed font (DejaVu, installed in the image) and fixed DPI
  mean two runs produce visually identical files.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless container; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from config import settings  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

LOG = get_logger(__name__)

#: Okabe-Ito qualitative palette, ordered for maximum separation of the first
#: few series, which is where most of these charts live.
PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#8C6D31",  # brown (extension, retains separation in greyscale)
    "#333333",  # near-black
)

#: Line styles cycled alongside colour so the figures survive monochrome print.
LINE_STYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)),
               (0, (1, 1)), (0, (4, 2, 1, 2)))
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

GRID_COLOUR = "#D9D9D9"
TEXT_COLOUR = "#1A1A1A"
MUTED_COLOUR = "#5A5A5A"


def apply_style() -> None:
    """Install the shared rcParams. Safe to call repeatedly."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": settings.FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": TEXT_COLOUR,
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID_COLOUR,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": TEXT_COLOUR,
        "ytick.color": TEXT_COLOUR,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "text.color": TEXT_COLOUR,
        "figure.autolayout": False,
    })


def series_style(index: int) -> dict[str, object]:
    """Colour, dash pattern and marker for the *n*th series in a chart."""
    return {
        "color": PALETTE[index % len(PALETTE)],
        "linestyle": LINE_STYLES[index % len(LINE_STYLES)],
        "marker": MARKERS[index % len(MARKERS)],
    }


def annotate_source(fig: Figure, source: str, note: str = "") -> None:
    """Stamp the data source, and any caveat, along the foot of the figure.

    Wraps by hand rather than relying on matplotlib's ``wrap=True``, which
    sizes itself against the pre-crop canvas and silently overruns the edge
    once ``savefig(bbox_inches="tight")`` crops a narrow (~8.5in) figure.
    """
    fontsize = 9
    avg_char_width_pt = 0.52 * fontsize
    usable_width_pt = fig.get_size_inches()[0] * 0.97 * 72
    chars_per_line = max(int(usable_width_pt / avg_char_width_pt), 20)
    wrapped_note = "\n".join(
        textwrap.fill(line, width=chars_per_line) for line in note.splitlines()
    ) if note else ""
    text = source if not wrapped_note else f"{source}\n{wrapped_note}"
    fig.text(
        0.005, -0.015, text, ha="left", va="top",
        fontsize=fontsize, color=MUTED_COLOUR,
    )


def finish(fig: Figure, path: Path, source: str, note: str = "") -> Path:
    """Attribute, save at publication DPI and close the figure."""
    annotate_source(fig, source, note)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=settings.FIGURE_FORMAT, dpi=settings.FIGURE_DPI)
    plt.close(fig)
    LOG.info("figure written: %s", path.name)
    return path


def thousands(value: float, _pos: int = 0) -> str:
    """Axis tick formatter: 1500000 -> '1.5M'."""
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= threshold:
            return f"{value / threshold:.1f}{suffix}"
    return f"{value:.0f}"
