"""H3 orthogonal decomposition and threshold sweep on Oxford-IIIT Pet (Critique 3.3)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.v22_common import (  # noqa: E402
    SEED,
    bootstrap_paired_ci,
    capture_per_head,
    load_spatial_names,
    measures_for_grid,
    resolve_data_root,
    sha256_file,
    verify_measures_grid14_equivalence,
)
from fastweight_vit.models import IMAGENET_PREPROCESSING, load_model  # noqa: E402
from fastweight_vit.models.dino import load_dino_model  # noqa: E402
from fastweight_vit.readings import capture  # noqa: E402
from fastweight_vit.tokens import centre_rings, high_norm_tokens  # noqa: E402

LAYERS = (9, 10, 11, 12)
THRESHOLDS = (1.5, 2.0, 2.5, 3.0, 4.0)
N_SHUFFLES = 5
METHODS = (
    "R0",
    "C3",
    "Z_out_3.0",
    "Soft_out_only",
    "NonOut_vw_zeroOut",
    "NonOut_vw_keepOut",
    "NonOut_ring_shuffle",
    "Z_out_1.5",
    "Z_out_2.0",
    "Z_out_2.5",
    "Z_out_4.0",
)


def nonoutlier_ring_perm(outlier_row: torch.Tensor, rings: torch.Tensor, layer: int, img_idx: int, draw: int) -> torch.Tensor:
    """Permute only non-outlier indices within each equal-radius ring; outliers stay fixed."""
    order = np.arange(196)
    non_out = (~outlier_row.cpu()).numpy()
    rings_np = rings.cpu().numpy()
    rng = np.random.default_rng(SEED * 1_000_003 + 10007 * draw + 1009 * img_idx + layer)
    for r in np.unique(rings_np):
        members = np.flatnonzero((rings_np == r) & non_out)
        if len(members) > 1:
            order[members] = rng.permutation(members)
    return torch.from_numpy(order).to(outlier_row.device)


def compute_scores_and_diags(store, img_indices: list[int], rings: torch.Tensor):
    device = store[LAYERS[0]]["received"].device
    batch = len(img_indices)
    scores = {m: torch.zeros((batch, 196), device=device) for m in METHODS if m != "NonOut_ring_shuffle"}
    shuffle_scores = [torch.zeros((batch, 196), device=device) for _ in range(N_SHUFFLES)]
    diag_records = []

    for layer in LAYERS:
        piece = store[layer]
        a = piece["received"]        # [B, H, 196]
        z = piece["length"]          # [B, H, 196]
        in_norms = piece["norms"]    # [B, 196]
        r0_layer = a.sum(dim=1)      # [B, 196]
        r0_total = r0_layer.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        for tau in THRESHOLDS:
            out_mask = high_norm_tokens(in_norms, ratio=tau)  # [B, 196]
            patch_frac = out_mask.float().mean(dim=-1)
            attn_share = (r0_layer * out_mask.float()).sum(dim=-1) / r0_total[:, 0]
            # Value norm ratio on outliers vs non-outlier median
            v_ratios = []
            for b_i in range(batch):
                if out_mask[b_i].any() and (~out_mask[b_i]).any():
                    med = z[b_i][:, ~out_mask[b_i]].median(dim=-1, keepdim=True).values.clamp_min(1e-12)
                    v_ratios.append(float((z[b_i][:, out_mask[b_i]] / med).mean()))
            diag_records.append({
                "layer": layer,
                "tau": tau,
                "patch_frac_sum": float(patch_frac.sum()),
                "attn_share_sum": float(attn_share.sum()),
                "n_with_outliers": len(v_ratios),
                "v_ratio_sum": float(sum(v_ratios)),
            })
            if tau == 3.0:
                out_3 = out_mask
            scores[f"Z_out_{tau:.1f}"] += r0_layer * (~out_mask).float()

        # Compute non-outlier median value norm per image and head [B, H, 1]
        med_z = torch.empty((batch, z.shape[1], 1), device=device, dtype=z.dtype)
        for b_i in range(batch):
            non_out = ~out_3[b_i]
            if non_out.any():
                med_z[b_i, :, 0] = z[b_i][:, non_out].median(dim=-1).values
            else:
                med_z[b_i, :, 0] = z[b_i].median(dim=-1).values

        o_b = out_3[:, None, :]  # [B, 1, 196]
        scores["R0"] += r0_layer
        scores["C3"] += (a * z).sum(dim=1)
        scores["Soft_out_only"] += (a * torch.where(o_b, z, med_z)).sum(dim=1)
        scores["NonOut_vw_zeroOut"] += (a * torch.where(o_b, torch.zeros_like(z), z)).sum(dim=1)
        scores["NonOut_vw_keepOut"] += (a * torch.where(o_b, med_z.expand_as(z), z)).sum(dim=1)

        for d in range(N_SHUFFLES):
            z_perm = torch.stack([
                z[b_i, :, nonoutlier_ring_perm(out_3[b_i], rings, layer, img_indices[b_i], d)]
                for b_i in range(batch)
            ])
            shuffle_scores[d] += (a * z_perm).sum(dim=1)

    return scores, shuffle_scores, diag_records


def load_eval_model(model_key: str, device: str):
    if model_key in ("deit", "deit_b"):
        return load_model(model_key, device=device), IMAGENET_PREPROCESSING[model_key]
    if model_key == "dino":
        # Backbone only; head is unused for attention maps and masks
        return load_dino_model(None, device=device), IMAGENET_PREPROCESSING["dino"]
    raise ValueError(f"Unknown model key {model_key}")


def evaluate_model(model_key: str, data_root: Path, device: str, batch_size: int, smoke: int):
    torch.backends.cuda.matmul.allow_tf32 = False
    model, pre = load_eval_model(model_key, device)
    names = load_spatial_names(smoke if smoke else None)
    dataset = __import__("experiments.v22_common", fromlist=["OxfordPetsSubset"]).OxfordPetsSubset(
        names, data_root, getattr(model, "backbone", model), patch_size=16, preprocessing=pre
    )
    rings = centre_rings(14, prefix=1)[1:].to(device)
    rows = []
    excluded = []
    agg_diag = {(l, t): {"patch_frac": 0.0, "attn_share": 0.0, "v_ratio_sum": 0.0, "n_out": 0}
                for l in LAYERS for t in THRESHOLDS}
    n_total_imgs = 0
    verified = False

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        idx_list = list(range(start, end))
        samples = [dataset[i] for i in idx_list]
        images = torch.stack([s[0] for s in samples]).to(device)
        fg = torch.stack([s[1] for s in samples])
        valid = torch.stack([s[2] for s in samples])
        other_fg = fg.roll(1, 0) if len(idx_list) > 1 else torch.zeros_like(fg)

        with torch.no_grad(), capture_per_head(model, LAYERS) as store:
            logits = model(images)

        if not verified:
            with torch.no_grad(), capture(model, LAYERS, per_head=True) as ref_store:
                ref_logits = model(images)
            assert torch.allclose(logits, ref_logits, rtol=3e-5, atol=1e-4)
            for l in LAYERS:
                r0_got = store[l]["received"].sum(dim=1)
                c3_got = (store[l]["received"] * store[l]["length"]).sum(dim=1)
                assert torch.allclose(r0_got, ref_store[l]["r0"], rtol=3e-5, atol=1e-4)
                assert torch.allclose(c3_got, ref_store[l]["c3"], rtol=3e-5, atol=1e-4)
                assert torch.allclose(store[l]["norms"], ref_store[l]["norms"], rtol=3e-5, atol=1e-4)
            verified = True

        scores, shuffle_scores, diag_recs = compute_scores_and_diags(store, idx_list, rings)
        n_total_imgs += len(idx_list)
        for rec in diag_recs:
            cell = agg_diag[(rec["layer"], rec["tau"])]
            cell["patch_frac"] += rec["patch_frac_sum"]
            cell["attn_share"] += rec["attn_share_sum"]
            cell["v_ratio_sum"] += rec["v_ratio_sum"]
            cell["n_out"] += rec["n_with_outliers"]

        if start == 0:
            verify_measures_grid14_equivalence(scores["C3"], fg, valid, other_fg, seed=SEED + start)

        # Evaluate deterministic methods
        per_m = {}
        for m, tensor in scores.items():
            per_m[m] = measures_for_grid(
                tensor, fg, valid, other_fg, grid=14, generator=torch.Generator().manual_seed(SEED + start)
            )
        # Evaluate 5 non-outlier ring-shuffles and average metrics per image
        shuf_list = [
            measures_for_grid(
                shuf_tensor, fg, valid, other_fg, grid=14,
                generator=torch.Generator().manual_seed(SEED + start + 100 * d),
            )
            for d, shuf_tensor in enumerate(shuffle_scores)
        ]

        for b_i, img_idx in enumerate(idx_list):
            if not (fg[b_i] & valid[b_i]).any() or not (valid[b_i] & ~fg[b_i]).any():
                excluded.append(names[img_idx])
                continue
            for m in scores:
                rows.append({
                    "model": model_key,
                    "index": img_idx,
                    "image": names[img_idx],
                    "method": m,
                    "enrichment": float(per_m[m]["enrichment"][b_i]),
                    "auroc": float(per_m[m]["auroc"][b_i]),
                    "precision": float(per_m[m]["precision"][b_i]),
                    "mass": float(per_m[m]["mass"][b_i]),
                })
            rows.append({
                "model": model_key,
                "index": img_idx,
                "image": names[img_idx],
                "method": "NonOut_ring_shuffle",
                "enrichment": float(np.mean([s["enrichment"][b_i] for s in shuf_list])),
                "auroc": float(np.mean([s["auroc"][b_i] for s in shuf_list])),
                "precision": float(np.mean([s["precision"][b_i] for s in shuf_list])),
                "mass": float(np.mean([s["mass"][b_i] for s in shuf_list])),
            })

    diag_rows = []
    for l in LAYERS:
        for t in THRESHOLDS:
            cell = agg_diag[(l, t)]
            diag_rows.append({
                "model": model_key,
                "layer": l,
                "tau": t,
                "patch_frac_pct": 100.0 * cell["patch_frac"] / n_total_imgs,
                "attn_share_pct": 100.0 * cell["attn_share"] / n_total_imgs,
                "mean_value_ratio": cell["v_ratio_sum"] / cell["n_out"] if cell["n_out"] else float("nan"),
                "images_with_outliers": cell["n_out"],
            })

    return rows, diag_rows, excluded


def summarize_model(rows: list[dict], model_key: str) -> dict:
    mrows = [r for r in rows if r["model"] == model_key]
    by_m = {m: {r["index"]: r for r in mrows if r["method"] == m} for m in METHODS}
    common_idx = sorted(set.intersection(*(set(v) for v in by_m.values())))
    out = {"n": len(common_idx), "methods": {}, "contrasts": {}}
    for metric in ("enrichment", "auroc"):
        out["methods"][metric] = {}
        vecs = {}
        for m in METHODS:
            vec = np.array([by_m[m][i][metric] for i in common_idx], dtype=np.float64)
            vecs[m] = vec
            diff_r0 = bootstrap_paired_ci(vec, vecs["R0"] if "R0" in vecs else vec) if m != "R0" else {"mean": 0.0, "ci95": [0.0, 0.0]}
            out["methods"][metric][m] = {
                "mean": float(vec.mean()),
                "delta_R0": diff_r0["mean"],
                "delta_R0_ci95": diff_r0["ci95"],
            }
        pairs = [
            ("C3_minus_R0", "C3", "R0"),
            ("Z_out_3.0_minus_R0", "Z_out_3.0", "R0"),
            ("C3_minus_Z_out_3.0", "C3", "Z_out_3.0"),
            ("Soft_out_only_minus_Z_out_3.0", "Soft_out_only", "Z_out_3.0"),
            ("NonOut_vw_zeroOut_minus_Z_out_3.0", "NonOut_vw_zeroOut", "Z_out_3.0"),
            ("C3_minus_NonOut_vw_zeroOut", "C3", "NonOut_vw_zeroOut"),
            ("C3_minus_NonOut_vw_keepOut", "C3", "NonOut_vw_keepOut"),
            ("C3_minus_NonOut_ring_shuffle", "C3", "NonOut_ring_shuffle"),
            ("C3_minus_Z_out_1.5", "C3", "Z_out_1.5"),
            ("C3_minus_Z_out_2.0", "C3", "Z_out_2.0"),
            ("C3_minus_Z_out_2.5", "C3", "Z_out_2.5"),
            ("C3_minus_Z_out_4.0", "C3", "Z_out_4.0"),
        ]
        out["contrasts"][metric] = {
            label: bootstrap_paired_ci(vecs[a], vecs[b]) for label, a, b in pairs
        }
        gain = (vecs["C3"] - vecs["R0"]).mean()
        if abs(gain) > 1e-6:
            out["contrasts"][metric]["decomposition_shares_of_C3_gain"] = {
                "Z_out_3.0_share": float((vecs["Z_out_3.0"] - vecs["R0"]).mean() / gain),
                "Soft_out_extra_over_Z_out": float((vecs["Soft_out_only"] - vecs["Z_out_3.0"]).mean() / gain),
                "NonOut_reweighting_extra_over_Z_out": float((vecs["NonOut_vw_zeroOut"] - vecs["Z_out_3.0"]).mean() / gain),
                "Remaining_joint_interaction_or_soft_outlier": float((vecs["C3"] - vecs["NonOut_vw_zeroOut"]).mean() / gain),
                "Spatial_placement_nonOut_share": float((vecs["C3"] - vecs["NonOut_ring_shuffle"]).mean() / gain),
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--models", nargs="+", default=["deit", "deit_b", "dino"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.perf_counter()
    data_root = resolve_data_root(args.data_root)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    all_rows, all_diags, excluded_map = [], [], {}
    for mk in args.models:
        print(f"[decomposition] Running {mk} on {args.device}...", flush=True)
        rows, diags, excl = evaluate_model(mk, data_root, args.device, args.batch_size, args.smoke)
        all_rows.extend(rows)
        all_diags.extend(diags)
        excluded_map[mk] = sorted(set(excl))

    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    with (out / "threshold_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_diags[0].keys()))
        w.writeheader()
        w.writerows(all_diags)

    summary = {
        "protocol_sha256": sha256_file(Path(__file__).resolve().parent / "protocol.md"),
        "smoke": args.smoke,
        "excluded": excluded_map,
        "elapsed_seconds": time.perf_counter() - t0,
        "models": {mk: summarize_model(all_rows, mk) for mk in args.models},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({mk: summary["models"][mk]["contrasts"]["enrichment"] for mk in args.models}, indent=2))


if __name__ == "__main__":
    main()
