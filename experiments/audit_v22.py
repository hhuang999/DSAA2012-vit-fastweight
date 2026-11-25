"""Local CPU verification & audit for all four v22 exploratory studies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.v22_common import bootstrap_mean_ci, bootstrap_paired_ci  # noqa: E402


def audit_decomposition() -> dict:
    d = ROOT / "experiments/decomposition"
    df = pd.read_csv(d / "per_image.csv")
    diag = pd.read_csv(d / "threshold_diagnostics.csv")
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    spatial = json.loads((ROOT / "experiments/spatial/summary.json").read_text(encoding="utf-8"))

    # Verify exact match with committed experiments/spatial on DeiT-S R0 and C3
    deit_s = summary["models"]["deit"]
    assert deit_s["n"] == spatial["n"] == 383
    assert abs(deit_s["methods"]["enrichment"]["R0"]["mean"] - spatial["methods"]["R0"]["enrichment"]["mean"]) < 1e-6
    assert abs(deit_s["methods"]["enrichment"]["C3"]["mean"] - spatial["methods"]["C3"]["enrichment"]["mean"]) < 1e-6
    assert abs(deit_s["methods"]["auroc"]["R0"]["mean"] - spatial["methods"]["R0"]["auroc"]["mean"]) < 1e-6
    assert abs(deit_s["methods"]["auroc"]["C3"]["mean"] - spatial["methods"]["C3"]["auroc"]["mean"]) < 1e-6

    # Verify bootstrap CI reproduction from per_image.csv
    sub = df[(df["model"] == "deit_b") & (df["method"].isin(["C3", "NonOut_vw_zeroOut", "Z_out_3.0"]))]
    piv = sub.pivot(index="index", columns="method", values="enrichment").dropna()
    ci = bootstrap_paired_ci(piv["NonOut_vw_zeroOut"].to_numpy(), piv["Z_out_3.0"].to_numpy())
    assert abs(ci["mean"] - summary["models"]["deit_b"]["contrasts"]["enrichment"]["NonOut_vw_zeroOut_minus_Z_out_3.0"]["mean"]) < 1e-10
    assert not diag.isna().sum().loc[["patch_frac_pct", "attn_share_pct"]].any()
    return {"passed": True, "deit_n": deit_s["n"], "rows": len(df)}


def audit_registers() -> dict:
    d = ROOT / "experiments/registers"
    df = pd.read_csv(d / "per_image.csv")
    diag = pd.read_csv(d / "diagnostics.csv")
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["models"]) == 5
    for mk, m in summary["models"].items():
        assert 375 <= m["n"] <= 384, (mk, m["n"])
        sub = df[(df["model"] == mk)]
        piv = sub.pivot(index="index", columns="method", values="enrichment").dropna()
        ci = bootstrap_paired_ci(piv["C3"].to_numpy(), piv["R0"].to_numpy())
        assert abs(ci["mean"] - m["contrasts"]["enrichment"]["C3_minus_R0"]["mean"]) < 1e-10
    assert len(diag) == 5 * 4
    return {"passed": True, "models": list(summary["models"].keys()), "rows": len(df)}


def audit_feature_origin() -> dict:
    d = ROOT / "experiments/feature_origin"
    df = pd.read_csv(d / "per_image.csv")
    cv = pd.read_csv(d / "construct_validity.csv")
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    for mk in ("deit", "deit_b"):
        meta = summary["meta"][mk]
        assert meta["layer1_self_q_lin_mean"] > 0.99, meta
        assert meta["layer1_self_q_sq_mean"] > 0.99, meta
        assert meta["hook_token_max_diff"] < 1e-4, meta
        sub = df[df["model"] == mk]
        piv = sub.pivot(index="index", columns="method", values="enrichment").dropna()
        ci = bootstrap_paired_ci(piv["C2_causal_Q"].to_numpy(), piv["R0"].to_numpy())
        assert abs(ci["mean"] - summary["relocation"][mk]["enrichment"]["contrasts"]["C2_causal_Q_minus_R0"]["mean"]) < 1e-10
    assert len(cv) > 0
    return {"passed": True, "reloc_rows": len(df), "cv_rows": len(cv)}


def audit_cls_deletion() -> dict:
    d = ROOT / "experiments/cls_deletion"
    df = pd.read_csv(d / "per_image.csv")
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    for mk in ("deit", "deit_b"):
        assert summary["models"][mk]["n"] == 128
        sub = df[df["model"] == mk]
        piv = sub.pivot(index="index", columns="method", values="srg_agree").dropna()
        assert len(piv) == 128
        ci = bootstrap_paired_ci(piv["cls_sum_head"].to_numpy(), piv["all_c3"].to_numpy())
        assert abs(ci["mean"] - summary["models"][mk]["endpoints"]["srg_agree"]["contrasts"]["cls_sum_head_minus_all_c3"]["mean"]) < 1e-10
    return {"passed": True, "rows": len(df)}


def main():
    report = {
        "decomposition": audit_decomposition(),
        "registers": audit_registers(),
        "feature_origin": audit_feature_origin(),
        "cls_deletion": audit_cls_deletion(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
