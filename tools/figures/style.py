"""The ONE shared style module for tools.figures (rcParams, palette, words, finalize, save).

Vendored palette tokens (copied as code, never imported from the NAS at run time):
  source: /mnt/ayp/agents/trm/26_figure-pipeline/11_lab-figure-pipeline/25_style/viz_style.py
  sha256: d6b7efccd7d0993343012d391c14704ab30a90a3b6b99330ebb21654c8ed29cb
rcParams follow section 1 of /home/ayp/project/trm/.claude/commands/fig-style.md.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.text import Text  # noqa: E402

# ----------------------------------------------------------------------------
# Vendored palette (viz_style.py, see header)
# ----------------------------------------------------------------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

DIVERGING_RED = "#a83232"
DIVERGING_MID = "#f0efec"
DIVERGING_BLUE = "#184f95"


def diverging_cmap():
    """test accuracy: 0 = fail (red) .. 0.5 = gray .. 1 = success (blue)."""
    return mcolors.LinearSegmentedColormap.from_list(
        "sigmak_diverging", [DIVERGING_RED, DIVERGING_MID, DIVERGING_BLUE], N=256
    )


SEQUENTIAL_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def sequential_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "sigmak_sequential", SEQUENTIAL_BLUE_STEPS, N=256
    )


STATUS_WARNING = "#fab219"
STATUS_WARNING_EDGE = "#9a7400"
STATUS_MUTED = "#c3c2b7"
STATUS_MUTED_LIGHT = "#e1e0d9"
STATUS_MUTED_EDGE = "#898781"


def text_color_for(rgba, threshold=0.6):
    r, g, b = rgba[0], rgba[1], rgba[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return INK_PRIMARY if lum > threshold else "#ffffff"


# One colour per data generation (built only from the vendored tokens above).
COHORT_COLORS = {
    "legacy": INK_MUTED,
    "clean_d0": DIVERGING_BLUE,
    "d1": SEQUENTIAL_BLUE_STEPS[1],
}

# ----------------------------------------------------------------------------
# rcParams
# ----------------------------------------------------------------------------
MIN_PRINT_PT = 7.0
_BASE_FONT_FAMILY = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
KO_FONT_FILE = "/home/ayp/.fonts/NotoSansCJKkr-Regular.otf"
KO_FONT_NAME = "Noto Sans CJK KR"

PUBLICATION_RCPARAMS = {
    "font.family": list(_BASE_FONT_FAMILY),
    "font.size": 16,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 2.5,
    "legend.frameon": False,
    "svg.fonttype": "none",
    # pilot additions: Type-42 fonts in PDF, white opaque background, ink tokens
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": INK_PRIMARY,
    "axes.labelcolor": INK_PRIMARY,
    "axes.edgecolor": INK_PRIMARY,
    "xtick.color": INK_PRIMARY,
    "ytick.color": INK_PRIMARY,
    "grid.color": GRID_HAIRLINE,
}

LANGS = ("en", "ko")


def apply_style(lang):
    """Apply the house style for lang in {"en", "ko"}. Call before creating figures."""
    if lang not in LANGS:
        raise ValueError(f"unknown lang {lang!r}; expected one of {LANGS}")
    params = dict(PUBLICATION_RCPARAMS)
    params["font.family"] = list(_BASE_FONT_FAMILY)
    if lang == "ko":
        font_manager.fontManager.addfont(KO_FONT_FILE)  # raises if the file is missing
        params["font.family"] = [KO_FONT_NAME] + list(_BASE_FONT_FAMILY)
        params["axes.unicode_minus"] = False
    else:
        params["axes.unicode_minus"] = True
    matplotlib.rcParams.update(params)


# ----------------------------------------------------------------------------
# Shared words
# ----------------------------------------------------------------------------
_WORDS = {
    "step": {"en": "step", "ko": "훈련 step"},
    "exact match": {"en": "exact match", "ko": "정확도"},
    "test": {"en": "test", "ko": "시험"},
    "train": {"en": "train", "ko": "훈련"},
    "budget": {"en": "budget", "ko": "예산"},
    "threshold 0.9": {"en": "threshold 0.9", "ko": "기준선 0.9"},
    "legacy": {"en": "legacy data", "ko": "옛 데이터"},
    "clean_d0": {"en": "clean data (d0)", "ko": "고친 데이터"},
    "d1": {"en": "second-generation data (d1)", "ko": "2세대 데이터"},
}


def word(lang, key):
    """Shared axis and legend word. Unknown lang or key raises KeyError."""
    if lang not in LANGS:
        raise KeyError(f"unknown lang {lang!r}")
    return _WORDS[key][lang]


# ----------------------------------------------------------------------------
# finalize / save
# ----------------------------------------------------------------------------
def finalize(fig, print_width_in):
    """Finishing pass and print-size check.

    scale = print_width_in / figure width (inches). Raises ValueError if any visible,
    non-empty text artist (labels, ticks, legend, annotations) would print below 7 pt.
    save() writes without bbox_inches="tight", so the PDF width equals the figure width.
    """
    if not print_width_in or print_width_in <= 0:
        raise ValueError(f"print_width_in must be positive, got {print_width_in!r}")
    if fig.get_layout_engine() is None:
        fig.tight_layout(pad=2)
    fig.canvas.draw()
    scale = float(print_width_in) / float(fig.get_figwidth())
    small = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        printed = t.get_fontsize() * scale
        if printed < MIN_PRINT_PT - 1e-9:
            small.append(f"{t.get_text()[:30]!r}: {printed:.2f} pt")
    if small:
        raise ValueError(
            f"text below {MIN_PRINT_PT:g} pt at print width {print_width_in} in "
            f"(scale {scale:.3f}): " + "; ".join(small[:8])
        )
    return scale


def save(fig, stem):
    """Write <stem>.png (dpi 300) and <stem>.pdf (no creation date). Returns both paths."""
    stem = str(stem)
    png, pdf = stem + ".png", stem + ".pdf"
    fig.savefig(png, dpi=300, facecolor="white", metadata={"Software": None})
    fig.savefig(pdf, facecolor="white", metadata={"CreationDate": None})
    return png, pdf


def close(fig):
    plt.close(fig)
