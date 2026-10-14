"""Figure style shared by the scripts: the conventions of jaxglitches' paper/validation/_style.py
(two-column widths, 8 pt text, Okabe-Ito colours, byte-deterministic PDFs), without its
manifest machinery."""
import datetime
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_IN = 3.31      # single column of the paper, 8.4 cm
FULL_IN = 6.85     # both columns, 17.4 cm

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})

# Okabe-Ito, in a fixed order; orange, purple and cyan sit below 3:1 contrast on white,
# so every figure pairs colour with a legend and a marker or line style.
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "cyan": "#56B4E9", "grey": "#666666", "black": "#000000"}
SERIES = [C["blue"], C["orange"], C["green"], C["red"], C["purple"], C["cyan"]]
MARKERS = ["o", "s", "^", "D", "v", "P"]

_EPOCH = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
_PDF_META = {"CreationDate": datetime.datetime.fromtimestamp(_EPOCH, datetime.timezone.utc)}


def save(fig, path):
    """Write `fig` to `path`; PDFs carry a pinned timestamp so reruns are byte-identical."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, metadata=_PDF_META if path.endswith(".pdf") else None)
    plt.close(fig)
    print(f"  wrote {path}")
