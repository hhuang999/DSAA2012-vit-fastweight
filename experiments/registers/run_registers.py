"""Cross-paradigm (CLIP, DINOv2) and register-trained control generalization (Critique 3.1)."""

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
import timm  # noqa: E402
import torch  # noqa: E402

from experiments.v22_common import (  # noqa: E402
    SEED,
    OxfordPetsSubset,
    bootstrap_paired_ci,
    capture_per_head,
    load_spatial_names,
    measures_for_grid,
    resolve_data_root,
    sha256_file,
)
from fastweight_vit.tokens import centre_rings, high_norm_tokens  # noqa: E402

MODELS = {
    "dinov2_s": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_s_reg4": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_b": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_b_reg4": "vit_base_patch14_reg4_dinov2.lvd142m",
    "clip_b": "vit_base_patch16_clip_224.openai",
}
LAYERS = (9, 10, 11, 12)
N_SHUFFLES = 5


def nonoutlier_ring_perm(outlier_row: torch.Tensor, rings: torch.Tensor, layer: int, img_idx: int, draw: int) -> torch.Tensor:
    n_patch = len(outlier_row)
    order = np.arange(n_patch)
    non_out = (~outlier_row.cpu()).numpy()
    rings_np = rings.cpu().numpy()
    rng = np.random.default_rng(SEED * 1_000_003 + 10007 * draw + 1009 * img_idx + layer)
    for r in np.unique(rings_np):
        members = np.flatnonzero((rings_np == r) & non_out)
        if len(members) > 1:
            order[members] = rng.permutation(members)
    return torch.from_numpy(order).to(outlier_row.device)


def evaluate_model(model_key: str, data_root: Path, device: str, batch_size: int, smoke: int):
    torch.backends.cuda.matmul.allow_tf32 = False
    timm_id = MODELS[model_key]
    model = timm.create_model(timm_id, pretrained=True).eval().to(device)
    patch_size = model.patch_embed.patch_size[0]
    prefix = model.num_prefix_tokens
    names = load_spatial_names(smoke if smoke else None)
    dataset = OxfordPetsSubset(names, data_root, model, patch_size=patch_size)
    grid = dataset.grid
    n_patches = grid * grid
    rings = centre_rings(grid, prefix=1)[1:].to(device)
    has_ls1 = any(hasattr(getattr(model.blocks[l - 1], "ls1", None), "gamma") for l in LAYERS)
    methods = ["R0", "C3", "Z_out_3.0", "Soft_out_only", "NonOut_vw_zeroOut", "NonOut_ring_shuffle"]
    if has_ls1:
        methods.append("C3_ls1")

    rows = []
    excluded = []
    agg_diag = {l: {"patch_frac": 0.0, "attn_share": 0.0, "v_ratio_sum": 0.0, "z_ratio_sum": 0.0, "n_out": 0}
                for l in LAYERS}
    n_total = 0
    verified = False

    # Use smaller batch for 518x518 DINOv2 (1369 tokens) to stay well within GPU RAM
    eff_bs = min(batch_size, 8) if grid > 20 else batch_size
    for start in range(0, len(dataset), eff_bs):
        end = min(start + eff_bs, len(dataset))
        idx_list = list(range(start, end))
        samples = [dataset[i] for i in idx_list]
        images = torch.stack([s[0] for s in samples]).to(device)
        fg = torch.stack([s[1] for s in samples])
        valid = torch.stack([s[2] for s in samples])
        other_fg = fg.roll(1, 0) if len(idx_list) > 1 else torch.zeros_like(fg)

        if not verified:
            with torch.no_grad():
                clean_logits = model(images)

        with torch.no_grad(), capture_per_head(model, LAYERS, prefix=prefix) as store:
            hooked_logits = model(images)

        if not verified:
            err = float((hooked_logits - clean_logits).abs().max())
            if err > 2e-4:
                raise AssertionError(f"{model_key} hooked forward mismatch: {err}")
            verified = True

        n_total += len(idx_list)
        scores = {m: torch.zeros((len(idx_list), n_patches), device=device)
                  for m in methods if m != "NonOut_ring_shuffle"}
        shuffle_scores = [torch.zeros((len(idx_list), n_patches), device=device) for _ in range(N_SHUFFLES)]

        for layer in LAYERS:
            piece = store[layer]
            a = piece["received"]        # [B, H, N]
            z = piece["length"]          # [B, H, N]
            vnorm = piece["value_norm"]  # [B, H, N]
            in_norms = piece["norms"]    # [B, N]
            r0_layer = a.sum(dim=1)
            out_3 = high_norm_tokens(in_norms, ratio=3.0)

            cell = agg_diag[layer]
            cell["patch_frac"] += float(out_3.float().mean(dim=-1).sum())
            cell["attn_share"] += float(((r0_layer * out_3.float()).sum(dim=-1) / r0_layer.sum(dim=-1).clamp_min(1e-12)).sum())
            for b_i in range(len(idx_list)):
                if out_3[b_i].any() and (~out_3[b_i]).any():
                    v_med = vnorm[b_i][:, ~out_3[b_i]].median(dim=-1, keepdim=True).values.clamp_min(1e-12)
                    z_med = z[b_i][:, ~out_3[b_i]].median(dim=-1, keepdim=True).values.clamp_min(1e-12)
                    cell["v_ratio_sum"] += float((vnorm[b_i][:, out_3[b_i]] / v_med).mean())
                    cell["z_ratio_sum"] += float((z[b_i][:, out_3[b_i]] / z_med).mean())
                    cell["n_out"] += 1

            med_z = torch.empty((len(idx_list), z.shape[1], 1), device=device, dtype=z.dtype)
            for b_i in range(len(idx_list)):
                non_out = ~out_3[b_i]
                med_z[b_i, :, 0] = z[b_i][:, non_out].median(dim=-1).values if non_out.any() else z[b_i].median(dim=-1).values

            o_b = out_3[:, None, :]
            scores["R0"] += r0_layer
            scores["C3"] += (a * z).sum(dim=1)
            scores["Z_out_3.0"] += r0_layer * (~out_3).float()
            scores["Soft_out_only"] += (a * torch.where(o_b, z, med_z)).sum(dim=1)
            scores["NonOut_vw_zeroOut"] += (a * torch.where(o_b, torch.zeros_like(z), z)).sum(dim=1)
            if has_ls1 and "length_ls1" in piece:
                scores["C3_ls1"] += (a * piece["length_ls1"]).sum(dim=1)

            for d in range(N_SHUFFLES):
                z_perm = torch.stack([
                    z[b_i, :, nonoutlier_ring_perm(out_3[b_i], rings, layer, idx_list[b_i], d)]
                    for b_i in range(len(idx_list))
                ])
                shuffle_scores[d] += (a * z_perm).sum(dim=1)

        per_m = {
            m: measures_for_grid(t, fg, valid, other_fg, grid=grid, generator=torch.Generator().manual_seed(SEED + start))
            for m, t in scores.items()
        }
        shuf_list = [
            measures_for_grid(st, fg, valid, other_fg, grid=grid, generator=torch.Generator().manual_seed(SEED + start + 100 * d))
            for d, st in enumerate(shuffle_scores)
        ]

        for b_i, img_idx in enumerate(idx_list):
            if not (fg[b_i] & valid[b_i]).any() or not (valid[b_i] & ~fg[b_i]).any():
                excluded.append(names[img_idx])
                continue
            for m in scores:
                rows.append({
                    "model": model_key, "index": img_idx, "image": names[img_idx], "method": m,
                    "enrichment": float(per_m[m]["enrichment"][b_i]),
                    "auroc": float(per_m[m]["auroc"][b_i]),
                    "precision": float(per_m[m]["precision"][b_i]),
                    "mass": float(per_m[m]["mass"][b_i]),
                })
            rows.append({
                "model": model_key, "index": img_idx, "image": names[img_idx], "method": "NonOut_ring_shuffle",
                "enrichment": float(np.mean([s["enrichment"][b_i] for s in shuf_list])),
                "auroc": float(np.mean([s["auroc"][b_i] for s in shuf_list])),
                "precision": float(np.mean([s["precision"][b_i] for s in shuf_list])),
                "mass": float(np.mean([s["mass"][b_i] for s in shuf_list])),
            })

    diag_rows = []
    for l in LAYERS:
        c = agg_diag[l]
        diag_rows.append({
            "model": model_key, "timm_id": timm_id, "grid": grid, "prefix_tokens": prefix, "layer": l,
            "patch_frac_pct": 100.0 * c["patch_frac"] / n_total,
            "attn_share_pct": 100.0 * c["attn_share"] / n_total,
            "mean_value_ratio": c["v_ratio_sum"] / c["n_out"] if c["n_out"] else float("nan"),
            "mean_proj_z_ratio": c["z_ratio_sum"] / c["n_out"] if c["n_out"] else float("nan"),
            "images_with_outliers": c["n_out"],
        })

    meta = {
        "timm_id": timm_id,
        "pretrained_cfg": model.pretrained_cfg,
        "grid": grid,
        "patch_size": patch_size,
        "prefix_tokens": prefix,
        "has_ls1": has_ls1,
        "excluded": excluded,
    }
    return rows, diag_rows, meta


def summarize_model(rows: list[dict], model_key: str) -> dict:
    mrows = [r for r in rows if r["model"] == model_key]
    methods = sorted({r["method"] for r in mrows})
    by_m = {m: {r["index"]: r for r in mrows if r["method"] == m} for m in methods}
    common_idx = sorted(set.intersection(*(set(v) for v in by_m.values())))
    out = {"n": len(common_idx), "methods": {}, "contrasts": {}}
    for metric in ("enrichment", "auroc"):
        vecs = {m: np.array([by_m[m][i][metric] for i in common_idx], dtype=np.float64) for m in methods}
        out["methods"][metric] = {
            m: {"mean": float(vecs[m].mean()),
                "delta_R0": bootstrap_paired_ci(vecs[m], vecs["R0"])["mean"],
                "delta_R0_ci95": bootstrap_paired_ci(vecs[m], vecs["R0"])["ci95"]}
            for m in methods
        }
        pairs = [
            ("C3_minus_R0", "C3", "R0"),
            ("Z_out_3.0_minus_R0", "Z_out_3.0", "R0"),
            ("C3_minus_Z_out_3.0", "C3", "Z_out_3.0"),
            ("Soft_out_only_minus_Z_out_3.0", "Soft_out_only", "Z_out_3.0"),
            ("NonOut_vw_zeroOut_minus_Z_out_3.0", "NonOut_vw_zeroOut", "Z_out_3.0"),
            ("C3_minus_NonOut_vw_zeroOut", "C3", "NonOut_vw_zeroOut"),
            ("C3_minus_NonOut_ring_shuffle", "C3", "NonOut_ring_shuffle"),
        ]
        if "C3_ls1" in vecs:
            pairs.append(("C3_ls1_minus_C3", "C3_ls1", "C3"))
        out["contrasts"][metric] = {
            label: bootstrap_paired_ci(vecs[a], vecs[b]) for label, a, b in pairs
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.perf_counter()
    data_root = resolve_data_root(args.data_root)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    all_rows, all_diags, metas = [], [], {}
    for mk in args.models:
        print(f"[registers] Running {mk} ({MODELS[mk]}) on {args.device}...", flush=True)
        rows, diags, meta = evaluate_model(mk, data_root, args.device, args.batch_size, args.smoke)
        all_rows.extend(rows)
        all_diags.extend(diags)
        metas[mk] = meta

    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    with (out / "diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_diags[0].keys()))
        w.writeheader()
        w.writerows(all_diags)

    summary = {
        "protocol_sha256": sha256_file(Path(__file__).resolve().parent / "protocol.md"),
        "smoke": args.smoke,
        "elapsed_seconds": time.perf_counter() - t0,
        "models": {mk: summarize_model(all_rows, mk) for mk in args.models},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(metas, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({mk: summary["models"][mk]["contrasts"]["enrichment"]["C3_minus_R0"] for mk in args.models}, indent=2))


if __name__ == "__main__":
    main()
