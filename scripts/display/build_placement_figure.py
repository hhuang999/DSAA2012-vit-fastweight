"""Render the horizontal placement-control enrichment chart from experiments/spatial/per_image.csv."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SEED = 2026
N_RESAMPLES = 10_000
ROWS = (
    ("R0", "Received attention", "#7C8B99"),
    ("C3", "Value weighting (native)", "#2F6685"),
    ("donor_rank", "Donor-image norm order", "#A3AEB8"),
    ("ring_shuffle", "Same-radius norm shuffle", "#A3AEB8"),
    ("norm_only", "Uniform attention, native norms", "#B8C1CA"),
)


def make_placement_figure(csv_path: Path, summary_path: Path, out: Path):
    table = pd.read_csv(csv_path)
    enrichment = table.pivot(index="index", columns="method", values="enrichment").dropna()
    if len(enrichment) != 383:
        raise ValueError(f"Expected 383 eligible pet images, got {len(enrichment)}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(enrichment), size=(N_RESAMPLES, len(enrichment)))

    means, lows, highs, labels, colors = [], [], [], [], []
    for method, label, color in ROWS:
        vals = enrichment[method].to_numpy()
        mean = float(vals.mean())
        expected = float(summary["methods"][method]["enrichment"]["mean"])
        if abs(mean - expected) > 1e-12:
            raise ValueError(f"{method} mean {mean} differs from summary.json {expected}")
        boot = vals[draws].mean(axis=1)
        means.append(mean)
        lows.append(mean - float(np.quantile(boot, 0.025)))
        highs.append(float(np.quantile(boot, 0.975)) - mean)
        labels.append(label)
        colors.append(color)

    for path in ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if Path(path).exists():
            matplotlib.font_manager.fontManager.addfont(path)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "pdf.fonttype": 42,
    })

    fig = plt.figure(figsize=(3.35, 1.32))
    ax = fig.add_axes([0.488, 0.25, 0.482, 0.70])
    y = np.arange(len(ROWS))[::-1]
    ax.axvline(means[0], color="#4D5B66", ls="--", lw=0.8, zorder=1)
    ax.barh(y, means, height=0.58, color=colors, edgecolor="none", zorder=2)
    ax.errorbar(means, y, xerr=[lows, highs], fmt="none", ecolor="#1F2933",
                elinewidth=0.8, capsize=1.8, zorder=3)
    for yi, m, hi in zip(y, means, highs):
        ax.text(m + hi + 0.004, yi, f"{m:.3f}", va="center", ha="left",
                fontsize=7.0, color="#1F2933",
                fontweight="bold" if yi == y[1] else "normal",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
                zorder=4)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(0.0, 0.205)
    ax.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlabel("Same-ring enrichment (383 pet crops)", labelpad=2, loc="right")
    ax.spines["left"].set_color("#8A96A3")

    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    if tight.x0 < -0.01 or tight.y0 < -0.01 or tight.x1 > w + 0.01 or tight.y1 > h + 0.01:
        raise ValueError(f"placement_controls: drawn content {tight} exceeds ({w} x {h} in)")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "placement_controls.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print("placement_controls: 3.35 x 1.32 in")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    make_placement_figure(
        ROOT / "experiments/spatial/per_image.csv",
        ROOT / "experiments/spatial/summary.json",
        args.out,
    )


if __name__ == "__main__":
    main()
