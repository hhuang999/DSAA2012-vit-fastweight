"""Build v22 2-panel main figure (Fig. 4) and generated appendix tables from committed v22 experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
OUT_FIG = ROOT / "paper/figures"
OUT_TAB = ROOT / "paper/tables"
SEED = 0
N_RESAMPLES = 1000


def configure_fonts():
    for path in (
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            matplotlib.font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6.8,
        "axes.labelsize": 6.8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def paired_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    diff = a - b
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(diff), size=(N_RESAMPLES, len(diff)))
    boot = diff[draws].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    mean = float(diff.mean())
    return mean, float(mean - lo), float(hi - mean)


def fmt_val(v: float, decimals: int = 3) -> str:
    s = f"{v:.{decimals}f}"
    return s.replace("-", "$-$")


def fmt_signed(v: float, decimals: int = 3) -> str:
    s = f"{v:+.{decimals}f}"
    return s.replace("-", "$-$")


def fmt_ci_dict(d: dict, decimals: int = 3, signed: bool = True) -> str:
    m = d["mean"]
    lo, hi = d["ci95"]
    if signed:
        s = f"{m:+.{decimals}f} [{lo:+.{decimals}f}, {hi:+.{decimals}f}]"
    else:
        s = f"{m:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"
    return s.replace("-", "$-$")


def write_table(name: str, cols: str, header: str, body: list[list[str]]):
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule", header + r" \\", r"\midrule"]
    for row in body:
        if row == ["\\midrule"]:
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    (OUT_TAB / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_mechanism_figure(out_dir: Path, keep_rc: bool = False):
    if not keep_rc:
        configure_fonts()
    dec = pd.read_csv(EXP / "decomposition/per_image.csv")
    reg = pd.read_csv(EXP / "registers/per_image.csv")

    # Panel A: Controlled ablations on DeiT-S & DeiT-B (enrichment delta vs R0)
    ablations = (
        ("C3", "Value weighting (full)"),
        ("Z_out_3.0", "Zero >3x outliers"),
        ("NonOut_vw_zeroOut", "Zero >3x + weight rest"),
        ("NonOut_ring_shuffle", "Shuffle non-outliers"),
        ("Z_out_1.5", "Zero >1.5x tail"),
    )
    panel_a = {}
    for mk in ("deit", "deit_b"):
        piv = dec[dec["model"] == mk].pivot(index="index", columns="method", values="enrichment").dropna()
        panel_a[mk] = [paired_ci(piv[m].to_numpy(), piv["R0"].to_numpy()) for m, _ in ablations]

    # Panel B: Cross-paradigm & register controls (AUROC: total C3-R0 vs spatial C3-NonOut_ring_shuffle)
    model_specs = (
        (dec, "deit", "DeiT-S"),
        (dec, "deit_b", "DeiT-B"),
        (reg, "clip_b", "CLIP ViT-B/16"),
        (reg, "dinov2_s", "DINOv2-S"),
        (reg, "dinov2_s_reg4", "DINOv2-S + reg4"),
        (reg, "dinov2_b", "DINOv2-B"),
        (reg, "dinov2_b_reg4", "DINOv2-B + reg4"),
        (dec, "dino", "DINO ViT-S"),
    )
    tot_b, spa_b = [], []
    for df_src, mk, _ in model_specs:
        piv = df_src[df_src["model"] == mk].pivot(index="index", columns="method", values="auroc").dropna()
        tot_b.append(paired_ci(piv["C3"].to_numpy(), piv["R0"].to_numpy()))
        spa_b.append(paired_ci(piv["C3"].to_numpy(), piv["NonOut_ring_shuffle"].to_numpy()))

    fig = plt.figure(figsize=(5.5, 1.62))
    ax1 = fig.add_axes([0.220, 0.24, 0.275, 0.62])
    ax2 = fig.add_axes([0.672, 0.24, 0.296, 0.62])

    # Plot Panel A
    y1 = np.arange(len(ablations))[::-1]
    h1 = 0.34
    ax1.axvline(0.0, color="#6E7B85", ls="--", lw=0.65, zorder=1)
    for offset, mk, label, color in (
        (+h1 / 2, "deit", "DeiT-S", "#2F6685"),
        (-h1 / 2, "deit_b", "DeiT-B", "#BB7D32"),
    ):
        vals = panel_a[mk]
        m = [v[0] for v in vals]
        lo = [v[1] for v in vals]
        hi = [v[2] for v in vals]
        ax1.barh(y1 + offset, m, height=h1, color=color, alpha=0.92, label=label, zorder=2)
        ax1.errorbar(m, y1 + offset, xerr=[lo, hi], fmt="none", ecolor="#1F2933",
                     elinewidth=0.65, capsize=1.4, zorder=3)
    ax1.set_yticks(y1, [lbl for _, lbl in ablations])
    ax1.tick_params(axis="y", length=0, pad=2.5)
    ax1.set_xlim(-0.015, 0.078)
    ax1.set_xticks([0.0, 0.02, 0.04, 0.06])
    ax1.set_xlabel(r"$\Delta$ Enrichment vs. received", labelpad=2)
    ax1.spines["left"].set_color("#8A96A3")
    ax1.legend(loc="upper right", bbox_to_anchor=(1.02, 1.02), frameon=False, ncol=1,
               fontsize=5.9, handlelength=1.1, handletextpad=0.4, labelspacing=0.25)
    ax1.set_title("a  Mechanism ablations (383 pet crops)", loc="left", fontsize=7.0, weight="bold", pad=6)

    # Plot Panel B
    y2 = np.arange(len(model_specs))[::-1]
    h2 = 0.34
    ax2.axvline(0.0, color="#6E7B85", ls="--", lw=0.65, zorder=1)
    for offset, series, label, color in (
        (+h2 / 2, tot_b, "Total gain (vs. received)", "#2F6685"),
        (-h2 / 2, spa_b, "Spatial gain (vs. shuffled)", "#BB7D32"),
    ):
        m = [v[0] for v in series]
        lo = [v[1] for v in series]
        hi = [v[2] for v in series]
        ax2.barh(y2 + offset, m, height=h2, color=color, alpha=0.92, label=label, zorder=2)
        ax2.errorbar(m, y2 + offset, xerr=[lo, hi], fmt="none", ecolor="#1F2933",
                     elinewidth=0.6, capsize=1.2, zorder=3)
    # Subtle separator before reg4 pairs and DINO reference
    ax2.axhline(4.5, color="#D8DDE2", lw=0.5, ls=":")
    ax2.axhline(0.5, color="#D8DDE2", lw=0.5, ls=":")
    ax2.set_yticks(y2, [lbl for _, _, lbl in model_specs])
    ax2.tick_params(axis="y", length=0, pad=2.5)
    ax2.set_xlim(-0.018, 0.082)
    ax2.set_xticks([0.0, 0.02, 0.04, 0.06, 0.08])
    ax2.set_xlabel(r"$\Delta$ Foreground AUROC", labelpad=2)
    ax2.spines["left"].set_color("#8A96A3")
    ax2.legend(loc="lower right", bbox_to_anchor=(1.02, 0.02), frameon=False, ncol=1,
               fontsize=5.4, handlelength=1.0, handletextpad=0.35, labelspacing=0.20)
    ax2.set_title("b  Pretraining & registers", loc="left", fontsize=6.9, weight="bold", pad=5)

    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    if tight.x0 < -0.01 or tight.y0 < -0.01 or tight.x1 > w + 0.01 or tight.y1 > h + 0.01:
        raise ValueError(f"mechanism_and_registers: drawn content {tight} exceeds ({w} x {h} in)")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "mechanism_and_registers.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print("mechanism_and_registers: 5.5 x 1.62 in")


def build_tables():
    # 1. v22_decomposition.tex
    dec = json.loads((EXP / "decomposition/summary.json").read_text(encoding="utf-8"))
    with (EXP / "decomposition/threshold_diagnostics.csv").open(encoding="utf-8", newline="") as f:
        tdiag = list(csv.DictReader(f))
    l12_diag = {(r["model"], float(r["tau"])): r for r in tdiag if int(r["layer"]) == 12}

    rows_dec = [
        ("R0", "Received attention"),
        ("Z_out_3.0", "Zero $>3\\times$ outliers"),
        ("Soft_out_only", "Soft $>3\\times$ outliers only"),
        ("NonOut_vw_zeroOut", "Zero $>3\\times$ + weight rest"),
        ("NonOut_ring_shuffle", "Shuffle non-outliers (ring)"),
        ("Z_out_2.0", "Zero $>2\\times$ patches"),
        ("Z_out_1.5", "Zero $>1.5\\times$ patches"),
        ("C3", "Value weighting (full)"),
    ]
    body1 = []
    for mk, mname in (("deit", "DeiT-S"), ("deit_b", "DeiT-B"), ("dino", "DINO ViT-S")):
        md = dec["models"][mk]["methods"]
        d3 = l12_diag[(mk, 3.0)]
        d15 = l12_diag[(mk, 1.5)]
        for idx, (mkey, mlabel) in enumerate(rows_dec):
            e = md["enrichment"][mkey]
            a = md["auroc"][mkey]
            model_col = mname if idx == 0 else ""
            if idx == 0:
                extra = f"L12 $>3\\times$: {float(d3['attn_share_pct']):.1f}\\% attn"
            elif idx == 1:
                extra = f"L12 $>1.5\\times$: {float(d15['attn_share_pct']):.1f}\\% attn"
            else:
                extra = ""
            body1.append([
                model_col,
                mlabel,
                fmt_val(e["mean"]),
                fmt_signed(e["delta_R0"]) if mkey != "R0" else "---",
                fmt_val(a["mean"]),
                fmt_signed(a["delta_R0"]) if mkey != "R0" else "---",
                extra,
            ])
        if mk != "dino":
            body1.append(["\\midrule"])
    write_table(
        "v22_decomposition.tex",
        "@{}llrrrrl@{}",
        r"Model & Readout & Enr. & $\Delta$Enr. & AUROC & $\Delta$AUROC & L12 outlier attention",
        body1,
    )

    # 2. v22_registers.tex
    reg = json.loads((EXP / "registers/summary.json").read_text(encoding="utf-8"))
    with (EXP / "registers/diagnostics.csv").open(encoding="utf-8", newline="") as f:
        rdiag = list(csv.DictReader(f))
    peak_attn = {}
    for mk in ("dinov2_s", "dinov2_s_reg4", "dinov2_b", "dinov2_b_reg4", "clip_b"):
        peak_attn[mk] = max(float(r["attn_share_pct"]) for r in rdiag if r["model"] == mk)
    reg_labels = (
        ("dinov2_s", "DINOv2-S ($37{\\times}37$)"),
        ("dinov2_s_reg4", "DINOv2-S + reg4 ($37{\\times}37$)"),
        ("dinov2_b", "DINOv2-B ($37{\\times}37$)"),
        ("dinov2_b_reg4", "DINOv2-B + reg4 ($37{\\times}37$)"),
        ("clip_b", "CLIP ViT-B/16 ($14{\\times}14$)"),
    )
    body2 = []
    for mk, lbl in reg_labels:
        md = reg["models"][mk]
        ce = md["contrasts"]["enrichment"]
        ca = md["contrasts"]["auroc"]
        body2.append([
            lbl,
            f"{peak_attn[mk]:.1f}\\%",
            fmt_ci_dict(ce["C3_minus_R0"]),
            fmt_ci_dict(ca["C3_minus_R0"]),
            fmt_ci_dict(ca["NonOut_vw_zeroOut_minus_Z_out_3.0"]),
            fmt_ci_dict(ca["C3_minus_NonOut_ring_shuffle"]),
        ])
    write_table(
        "v22_registers.tex",
        "@{}lrrrrr@{}",
        r"Model (grid) & Peak $>3\times$ attn & $\Delta$Enr. ($C_3{-}R_0$) & $\Delta$AUROC ($C_3{-}R_0$) & $\Delta$AUROC (non-out.) & $\Delta$AUROC (spatial)",
        body2,
    )

    # 3. v22_feature_origin.tex
    fo = json.loads((EXP / "feature_origin/summary.json").read_text(encoding="utf-8"))
    body3 = []
    for mk, mname in (("deit", "DeiT-S"), ("deit_b", "DeiT-B")):
        cv = fo["construct_validity"][mk]
        rel = fo["relocation"][mk]
        for sub, slabel in (("background", "All background"), ("high_norm_bg", "High-norm bg")):
            s = cv[sub]
            body3.append([
                mname if sub == "background" else "",
                slabel,
                f"{s['P_self']['mean']:.3f} / {s['Q_self']['mean']:.3f}",
                f"{s['P_euclid_disp']['mean']:.2f} / {s['Q_euclid_disp']['mean']:.2f}",
                f"{s['P_fg_mass']['mean']:.3f} / {s['Q_fg_mass']['mean']:.3f}",
                fmt_ci_dict(rel["enrichment"]["contrasts"]["C2_causal_Q_minus_R0"]) if sub == "background" else fmt_ci_dict(rel["enrichment"]["contrasts"]["C2_probe_minus_R0"]),
                fmt_ci_dict(rel["auroc"]["contrasts"]["C2_causal_Q_minus_R0"]) if sub == "background" else fmt_ci_dict(rel["auroc"]["contrasts"]["C2_probe_minus_R0"]),
            ])
        if mk == "deit":
            body3.append(["\\midrule"])
    # Wait: make row labels explicit so causal Q vs probe relocation isn't conflated with subset rows!
    body3_clean = []
    for mk, mname in (("deit", "DeiT-S"), ("deit_b", "DeiT-B")):
        cv = fo["construct_validity"][mk]
        rel = fo["relocation"][mk]
        bg = cv["background"]
        hn = cv["high_norm_bg"]
        body3_clean.append([
            mname,
            "Bg ($P$ / $Q$ / $Q^{\\mathrm{sq}}$)",
            f"{bg['P_self']['mean']:.3f} / {bg['Q_self']['mean']:.3f} / {bg['Qsq_self']['mean']:.3f}",
            f"{bg['P_euclid_disp']['mean']:.2f} / {bg['Q_euclid_disp']['mean']:.2f} / {bg['Qsq_euclid_disp']['mean']:.2f}",
            f"{bg['P_fg_mass']['mean']:.3f} / {bg['Q_fg_mass']['mean']:.3f}",
            f"Causal $Q$: {fmt_ci_dict(rel['enrichment']['contrasts']['C2_causal_Q_minus_R0'])}",
            f"Causal $Q$: {fmt_ci_dict(rel['auroc']['contrasts']['C2_causal_Q_minus_R0'])}",
        ])
        body3_clean.append([
            "",
            "High-norm bg ($P$ / $Q$ / $Q^{\\mathrm{sq}}$)",
            f"{hn['P_self']['mean']:.3f} / {hn['Q_self']['mean']:.3f} / {hn['Qsq_self']['mean']:.3f}",
            f"{hn['P_euclid_disp']['mean']:.2f} / {hn['Q_euclid_disp']['mean']:.2f} / {hn['Qsq_euclid_disp']['mean']:.2f}",
            f"{hn['P_fg_mass']['mean']:.3f} / {hn['Q_fg_mass']['mean']:.3f}",
            f"Probe $P$: {fmt_ci_dict(rel['enrichment']['contrasts']['C2_probe_minus_R0'])}",
            f"Probe $P$: {fmt_ci_dict(rel['auroc']['contrasts']['C2_probe_minus_R0'])}",
        ])
        body3_clean.append([
            "",
            "All patches (ref $C_3$ vs. $Q^{\\mathrm{sq}}$)",
            f"{cv['all']['P_self']['mean']:.3f} / {cv['all']['Q_self']['mean']:.3f} / {cv['all']['Qsq_self']['mean']:.3f}",
            f"{cv['all']['P_euclid_disp']['mean']:.2f} / {cv['all']['Q_euclid_disp']['mean']:.2f} / {cv['all']['Qsq_euclid_disp']['mean']:.2f}",
            f"{cv['all']['P_fg_mass']['mean']:.3f} / {cv['all']['Q_fg_mass']['mean']:.3f}",
            f"$Q^{{\\mathrm{{sq}}}}$: {fmt_ci_dict(rel['enrichment']['contrasts']['C2_causal_Qsq_minus_R0'])}",
            f"$C_3$: {fmt_ci_dict(rel['enrichment']['contrasts']['C3_minus_R0'])}",
        ])
        if mk == "deit":
            body3_clean.append(["\\midrule"])
    write_table(
        "v22_feature_origin.tex",
        "@{}llcccrr@{}",
        r"Model & Token subset & Self-mass & Mean disp. (patches) & Fg mass ($P$/$Q$) & L12 $\Delta$Enr. ($-R_0$) & L12 $\Delta$AUROC / ref",
        body3_clean,
    )

    # 4. v22_cls_deletion.tex
    cd = json.loads((EXP / "cls_deletion/summary.json").read_text(encoding="utf-8"))
    rows_cd = (
        ("cls_sum_head_minus_all_c3", "CLS value weighting $-$ all-query $C_3$"),
        ("cls_net_minus_all_c3", "CLS net contribution $-$ all-query $C_3$"),
        ("cls_net_minus_cls_sum_head", "CLS net contribution $-$ CLS value weighting"),
        ("cls_net_minus_all_r0", "CLS net contribution $-$ all-query $R_0$"),
        ("all_c3_minus_all_r0", "All-query $C_3$ $-$ all-query $R_0$"),
    )
    body4 = []
    for mk, mname in (("deit", "DeiT-S"), ("deit_b", "DeiT-B")):
        eps = cd["models"][mk]["endpoints"]
        for idx, (ck, clabel) in enumerate(rows_cd):
            body4.append([
                mname if idx == 0 else "",
                clabel,
                fmt_ci_dict(eps["srg_agree"]["contrasts"][ck]),
                fmt_ci_dict(eps["prob_sep"]["contrasts"][ck]),
                fmt_ci_dict(eps["kl_sep"]["contrasts"][ck]),
            ])
        if mk == "deit":
            body4.append(["\\midrule"])
    write_table(
        "v22_cls_deletion.tex",
        "@{}llrrr@{}",
        r"Model & Paired layer-12 contrast & Binary SRG [95\% CI] & Prob. sep. [95\% CI] & KL sep. [95\% CI]",
        body4,
    )
    print("v22_tables: v22_decomposition, v22_registers, v22_feature_origin, v22_cls_deletion")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-fig", type=Path, default=OUT_FIG)
    args = parser.parse_args()
    build_mechanism_figure(args.out_fig)
    build_tables()


if __name__ == "__main__":
    main()
