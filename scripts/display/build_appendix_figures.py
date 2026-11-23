"""Build the appendix figures added in the final report (Figs. A.1, B.2 and C.1)."""

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"deit": "#2F6685", "deit_b": "#BB7D32", "dino": "#7B6A90"}
STYLES = {"deit": "-", "deit_b": "--", "dino": ":"}
NAMES = {"deit": "DeiT-S", "deit_b": "DeiT-B", "dino": "DINO ViT-S"}
GREY = "#929292"
PANEL = "#F0F0EE"


def configure_fonts():
    """Register Liberation Sans on Linux runners and match manuscript typography."""
    for path in ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if Path(path).exists():
            matplotlib.font_manager.fontManager.addfont(path)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 7.2, "axes.labelsize": 7.2, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.size": 2.5,
        "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def title(ax, letter, text):
    ax.set_title(f"{letter}  {text}", loc="left", fontsize=7.5, weight="bold", pad=4)


def metrics_schematic(out):
    """Fig. A.1: schematic of same-ring enrichment, ClickMe alignment and deletion score."""
    fig = plt.figure(figsize=(5.5, 1.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.05], wspace=0.42,
                          left=0.02, right=0.99, top=0.86, bottom=0.22)

    # (a) same-ring enrichment on a 14x14 grid
    ax = fig.add_subplot(gs[0])
    n = 14
    yy, xx = np.mgrid[0:n, 0:n]
    centre = (n - 1) / 2
    radius = np.hypot(xx - centre, yy - centre)
    fg = ((xx - 9.2) ** 2 / 9 + (yy - 4.6) ** 2 / 6) < 1
    rng = np.random.default_rng(0)
    score = 0.6 * fg + 0.5 * np.exp(-radius ** 2 / 30) + 0.25 * rng.random((n, n))
    k = int(fg.sum())
    top = np.zeros(n * n, bool)
    top[np.argsort(-score.ravel())[:k]] = True
    top = top.reshape(n, n)
    for i in range(n):
        for j in range(n):
            face = "#C3D4E2" if fg[i, j] else "white"
            ax.add_patch(Rectangle((j, i), 1, 1, fc=face, ec="#D0D0D0", lw=0.3))
            if top[i, j]:
                ax.plot(j + 0.5, i + 0.5, "o", ms=2.2, color=COLORS["deit"])
    for r in (2.5, 4.5, 6.5):
        ax.add_patch(plt.Circle((centre + 0.5, centre + 0.5), r, fill=False, ec=GREY,
                                lw=0.6, ls=(0, (2, 1.5))))
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    title(ax, "a", "Same-ring enrichment")
    ax.text(0, n + 1.6, "shaded: object; dots: top-$k$ patches;\ndashed: equal-radius rings",
            fontsize=6.2, va="top", color="#444444")

    # (b) ClickMe alignment as a difference of two correlations
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    title(ax, "b", "ClickMe alignment")
    boxes = [(0.2, 2.2, "readout\nmap of\nimage $x$"), (6.2, 3.7, "human map\nof $x$"),
             (6.2, 0.6, "human map of\nanother\nimage")]
    for x, y, label in boxes:
        ax.add_patch(Rectangle((x, y), 3.4 if x > 1 else 2.6, 1.6, fc=PANEL, ec="#B5B5B5", lw=0.5))
        ax.text(x + (1.7 if x > 1 else 1.3), y + 0.8, label, ha="center", va="center", fontsize=6.3)
    ax.annotate("", xy=(6.15, 4.5), xytext=(2.85, 3.2),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=COLORS["deit"]))
    ax.annotate("", xy=(6.15, 1.4), xytext=(2.85, 2.8),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=GREY))
    ax.text(4.4, 4.45, r"$\rho_{\mathrm{own}}$", fontsize=7, color=COLORS["deit"], ha="center")
    ax.text(4.4, 1.25, r"$\rho_{\mathrm{other}}$", fontsize=7, color="#555555", ha="center")
    ax.text(5.0, -0.55, r"alignment $=\rho_{\mathrm{own}}-\rho_{\mathrm{other}}$",
            ha="center", fontsize=6.6)

    # (c) deletion score as the area between two retention curves
    ax = fig.add_subplot(gs[2])
    f = np.linspace(0, 0.75, 120)
    lif = 1 - 0.25 * (f / 0.75) ** 2.2
    mif = np.exp(-f / 0.16) * 0.92 + 0.08
    ax.plot(f, lif, color=COLORS["deit"], lw=1.0)
    ax.plot(f, mif, color=GREY, lw=1.0, ls="--")
    ax.fill_between(f, mif, lif, color=COLORS["deit"], alpha=0.13, lw=0)
    ax.text(0.73, 0.70, "least important first", fontsize=6.1, color=COLORS["deit"], ha="right", va="top")
    ax.text(0.73, 0.22, "most important\nfirst", fontsize=6.1, color="#555555", ha="right", va="bottom",
            linespacing=1.1)
    ax.text(0.19, 0.50, "deletion\nscore", fontsize=6.4, ha="center")
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fraction of patches removed")
    ax.set_ylabel("Top-1 retained")
    title(ax, "c", "Deletion score")
    fig.savefig(out / "metric_schematic.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def value_norm_and_recovery(results, out):
    """Fig. B.2: value-norm ratio of high-norm patches per layer; layer-12 recovery."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.75), gridspec_kw=dict(width_ratios=[1.15, 1]))
    plt.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.22, wspace=0.34)

    ax = axes[0]
    data = rows(results / "value_norm.csv")
    for model in ("deit", "deit_b"):
        pts = [(int(r["layer"]), float(r["value_ratio"])) for r in data
               if r["model"] == model and r["layer"].isdigit() and r["value_ratio"]]
        pts.sort()
        x, y = zip(*pts)
        ax.plot(x, y, color=COLORS[model], ls=STYLES[model], lw=1.05, marker="o", ms=2.2,
                label=NAMES[model])
    ax.axhline(1, color=GREY, lw=0.6)
    ax.axvspan(8.5, 12.5, color=PANEL, lw=0, zorder=0)
    ax.set_xticks([1, 4, 8, 12])
    ax.set_xlim(0.5, 12.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Ratio to median value norm")
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.8)
    title(ax, "a", "Value norms of high-norm patches")

    ax = axes[1]
    rec = rows(results / "last_layer.csv")
    labels = {"enrichment": "Enrichment", "clickme": "ClickMe", "srg": "Deletion"}
    order = ["enrichment", "clickme", "srg"]
    width = 0.36
    for k, model in enumerate(("deit", "deit_b")):
        for m, metric in enumerate(order):
            r = next(r for r in rec if r["model"] == model and r["metric"] == metric and r["reading"] == "C3")
            mean, low, high = (100 * float(r[c]) for c in ("recovered", "low", "high"))
            x = m + (k - 0.5) * width
            ax.bar(x, mean, width * 0.9, color=COLORS[model], alpha=0.85 if k == 0 else 0.6,
                   label=NAMES[model] if m == 0 else None)
            ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], color="#333333", lw=0.6,
                        capsize=1.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([labels[m] for m in order])
    ax.set_ylabel("Layer-12 drop recovered (%)")
    ax.set_ylim(0, 70)
    ax.legend(loc="upper right", fontsize=6.5)
    title(ax, "b", "Layer-12 recovery by value weighting")
    fig.savefig(out / "value_norm_recovery.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def probe_errors(results, out):
    """Fig. C.1: locality of probe errors and linear versus MLP probe accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.7), gridspec_kw=dict(width_ratios=[1.15, 1]))
    plt.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.22, wspace=0.34)

    ax = axes[0]
    err = rows(results / "c2_analysis" / "errors.csv")
    for model in ("deit", "deit_b", "dino"):
        pts = sorted((int(r["layer"]), 100 * float(r["mean"])) for r in err
                     if r["model"] == model and r["source"] == "probe" and r["metric"] == "near3x3")
        x, y = zip(*pts)
        ax.plot(x, y, color=COLORS[model], ls=STYLES[model], lw=1.05, marker="o", ms=2.2,
                label=NAMES[model])
    ax.axvspan(8.5, 12.5, color=PANEL, lw=0, zorder=0)
    ax.set_xlim(5.5, 12.5)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mass within 3x3 of true slot (%)")
    ax.legend(loc="lower left", fontsize=6.5, handlelength=1.8)
    title(ax, "a", "Locality of probe errors")

    ax = axes[1]
    mlp = rows(results / "supplementary" / "mlp_probe.csv")
    width = 0.36
    for m, model in enumerate(("deit", "deit_b", "dino")):
        for k, probe in enumerate(("linear", "mlp")):
            r = next(r for r in mlp if r["model"] == model and r["layer"] == "12"
                     and r["probe"] == probe and r["quantity"] == "exact")
            ax.bar(m + (k - 0.5) * width, 100 * float(r["mean"]), width * 0.9, color=COLORS[model],
                   alpha=0.85 if k == 0 else 0.45, hatch=None if k == 0 else "////",
                   edgecolor="white", lw=0.3)
    ax.axhline(100 / 196, color=GREY, lw=0.6, ls="--")
    ax.set_xticks(range(3))
    ax.set_xticklabels([NAMES[m] for m in ("deit", "deit_b", "dino")])
    ax.set_ylabel("Layer-12 probe accuracy (%)")
    ax.set_ylim(0, 50)
    handles = [Rectangle((0, 0), 1, 1, fc="#777777", alpha=0.85),
               Rectangle((0, 0), 1, 1, fc="#777777", alpha=0.45, hatch="////", ec="white"),
               Line2D([], [], color=GREY, ls="--", lw=0.6)]
    ax.legend(handles, ["linear", "MLP", "chance"], loc="upper left", fontsize=6.5, ncol=3,
              handlelength=1.4, columnspacing=0.8)
    title(ax, "b", "Linear and MLP probes")
    fig.savefig(out / "probe_errors.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def build_all(results: Path, out: Path):
    configure_fonts()
    out.mkdir(parents=True, exist_ok=True)
    metrics_schematic(out)
    value_norm_and_recovery(results, out)
    probe_errors(results, out)
    print("appendix_figures: metric_schematic, value_norm_recovery, probe_errors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    build_all(args.results, args.out)


if __name__ == "__main__":
    main()
