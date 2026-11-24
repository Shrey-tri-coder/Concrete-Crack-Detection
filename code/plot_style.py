"""Consistent matplotlib styling for all report figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
CLASS_COL = {0: BLUE, 1: ORANGE}
CLASS_NAME = {0: "Uncracked", 1: "Cracked"}
METHOD_COL = {"Logistic Regression": BLUE, "Random Forest": ORANGE, "CNN": AQUA}
SURF_COL = {"D": BLUE, "W": ORANGE, "P": AQUA}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "lines.linewidth": 2, "axes.facecolor": SURFACE,
})


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("saved", path.name)
