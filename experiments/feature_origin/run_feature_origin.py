"""Causal input-occlusion feature origin Q(p | x_j) vs linear position probe P(p | x_j) (Critique 3.2)."""

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
    OxfordPetsSubset,
    bootstrap_mean_ci,
    bootstrap_paired_ci,
    capture_per_head,
    load_cls_names,
    measures_for_grid,
    resolve_data_root,
    sha256_file,
)
from fastweight_vit.models import IMAGENET_PREPROCESSING, load_model  # noqa: E402
from fastweight_vit.readings import (  # noqa: E402
    gaussian_relocation,
    load_probes,
    match_sigma,
    relocate,
    squared_distances,
)
from fastweight_vit.tokens import high_norm_tokens  # noqa: E402

PATCH_SIZE = 16
GRID = 14
N_PATCHES = 196


def forward_upto_norm1(model, images: torch.Tensor, block_idx: int) -> torch.Tensor:
    """Return normalised patch tokens x_j fed into block `block_idx` attention [B, 196, D]."""
    x = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(images))))
    for b in range(block_idx):
        x = model.blocks[b](x)
    return model.blocks[block_idx].norm1(x)[:, 1:]


def occluded_variants(image: torch.Tensor) -> torch.Tensor:
    """Create [196, 3, 224, 224] variants with each 16x16 patch p replaced by 0.0 (ImageNet mean)."""
    occ = image.unsqueeze(0).repeat(N_PATCHES, 1, 1, 1)
    for p in range(N_PATCHES):
        r, c = divmod(p, GRID)
        occ[p, :, r * PATCH_SIZE:(r + 1) * PATCH_SIZE, c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = 0.0
    return occ


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise Jensen-Shannon divergence (base e, [0, ln 2]) between [196, 196] distributions."""
    p_c = p.clamp_min(eps)
    q_c = q.clamp_min(eps)
    m = 0.5 * (p_c + q_c)
    kl_pm = (p_c * (p_c.log() - m.log())).sum(-1)
    kl_qm = (q_c * (q_c.log() - m.log())).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def compute_subset_stats(name: str, mask: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, Q_sq: torch.Tensor,
                         fg: torch.Tensor, dist: torch.Tensor) -> dict | None:
    if not mask.any():
        return None
    idx = mask
    fg_f = fg.float()
    self_idx = torch.arange(N_PATCHES, device=P.device)
    p_self = P[self_idx, self_idx][idx].mean()
    q_self = Q[self_idx, self_idx][idx].mean()
    qsq_self = Q_sq[self_idx, self_idx][idx].mean()
    p_disp = (P * dist).sum(-1)[idx].mean()
    q_disp = (Q * dist).sum(-1)[idx].mean()
    qsq_disp = (Q_sq * dist).sum(-1)[idx].mean()
    jsd = js_divergence(P, Q)[idx].mean()
    p_fg_mass = (P @ fg_f)[idx].mean()
    q_fg_mass = (Q @ fg_f)[idx].mean()
    qsq_fg_mass = (Q_sq @ fg_f)[idx].mean()
    p_arg_fg = fg_f[P.argmax(-1)][idx].mean()
    q_arg_fg = fg_f[Q.argmax(-1)][idx].mean()
    return {
        "subset": name,
        "count": int(idx.sum()),
        "P_self": float(p_self),
        "Q_self": float(q_self),
        "Qsq_self": float(qsq_self),
        "P_euclid_disp": float(p_disp),
        "Q_euclid_disp": float(q_disp),
        "Qsq_euclid_disp": float(qsq_disp),
        "JSD_P_Q": float(jsd),
        "P_fg_mass": float(p_fg_mass),
        "Q_fg_mass": float(q_fg_mass),
        "Qsq_fg_mass": float(qsq_fg_mass),
        "P_argmax_fg_rate": float(p_arg_fg),
        "Q_argmax_fg_rate": float(q_arg_fg),
    }


def evaluate_model(model_key: str, data_root: Path, device: str, occ_batch: int, smoke: int):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(model_key, device=device)
    n_imgs = smoke if smoke else 64
    names = load_cls_names(n_imgs)
    saved_names = (Path(__file__).resolve().parent / "selected_names.txt").read_text(encoding="utf-8").splitlines()
    if not smoke and names != saved_names:
        raise ValueError("Selected names mismatch archived experiments/feature_origin/selected_names.txt")
    dataset = OxfordPetsSubset(names, data_root, model, patch_size=16, preprocessing=IMAGENET_PREPROCESSING[model_key])
    probes, sigmas = load_probes(ROOT / f"outputs/diagnose_probe/{model_key}/probes.npz")
    probe12 = probes[12].to(device)
    sq_dist = squared_distances(GRID, device=device)
    euclid_dist = sq_dist.sqrt()

    # Layer-1 sanity check on first 4 images
    l1_self_lin, l1_self_sq = [], []
    for i in range(min(4, len(dataset))):
        img, _, _, _ = dataset[i]
        img_dev = img.unsqueeze(0).to(device)
        x1_orig = forward_upto_norm1(model, img_dev, block_idx=0)[0]  # [196, D]
        occ = occluded_variants(img).to(device)
        x1_occ = forward_upto_norm1(model, occ, block_idx=0)  # [196_occ, 196_tok, D]
        delta = (x1_occ - x1_orig.unsqueeze(0)).norm(dim=-1).T  # [196_tok, 196_occ]
        q1 = delta / delta.sum(-1, keepdim=True).clamp_min(1e-12)
        d2 = delta ** 2
        q1_sq = d2 / d2.sum(-1, keepdim=True).clamp_min(1e-12)
        l1_self_lin.append(float(q1.diagonal().mean()))
        l1_self_sq.append(float(q1_sq.diagonal().mean()))
    print(f"[feature_origin] {model_key} Layer-1 self Q_lin={np.mean(l1_self_lin):.4f}, Q_sq={np.mean(l1_self_sq):.4f}", flush=True)
    if np.mean(l1_self_sq) < 0.98:
        raise AssertionError(f"Layer-1 Q_sq sanity check failed: {l1_self_sq}")

    reloc_rows, cv_rows = [], []
    excluded = []
    # First pass collect all P, Q, Qsq, r0, c3 so blur_Q uses mean squared displacement across evaluated cohort (or matched per-image / cohort)
    cached_items = []
    msd_q_list, msd_qsq_list = [], []
    hook_err_max = 0.0

    for i in range(len(dataset)):
        img, fg, valid, _ = dataset[i]
        img_dev = img.unsqueeze(0).to(device)
        with torch.no_grad(), capture_per_head(model, (12,)) as store:
            _ = model(img_dev)
        x12_orig = forward_upto_norm1(model, img_dev, block_idx=11)[0]
        if i == 0:
            # Verify forward_upto_norm1 matches hooked tokens
            # capture_per_head doesn't store tokens, so compare via fastweight_vit.readings.capture
            from fastweight_vit.readings import capture as ref_cap
            with torch.no_grad(), ref_cap(model, (12,)) as rstore:
                _ = model(img_dev)
            hook_err_max = float((x12_orig - rstore[12]["tokens"][0]).abs().max())
            if hook_err_max > 1e-4:
                raise AssertionError(f"x12 token mismatch: {hook_err_max}")

        s_j = store[12]["received"].sum(dim=1)[0]              # [196]
        c3_j = (store[12]["received"] * store[12]["length"]).sum(dim=1)[0]
        in_norm = store[12]["norms"][0]                        # [196]
        P = probe12.probs(x12_orig)                            # [196_tok, 196_pos]

        occ = occluded_variants(img).to(device)
        chunks = []
        for s0 in range(0, N_PATCHES, occ_batch):
            chunks.append(forward_upto_norm1(model, occ[s0:s0 + occ_batch], block_idx=11))
        x12_occ = torch.cat(chunks, dim=0)                     # [196_occ, 196_tok, D]
        delta = (x12_occ - x12_orig.unsqueeze(0)).norm(dim=-1).T  # [196_tok, 196_occ]
        Q = delta / delta.sum(-1, keepdim=True).clamp_min(1e-12)
        d2 = delta ** 2
        Q_sq = d2 / d2.sum(-1, keepdim=True).clamp_min(1e-12)

        msd_q_list.append(float((Q * sq_dist).sum(-1).mean()))
        msd_qsq_list.append(float((Q_sq * sq_dist).sum(-1).mean()))

        fg_dev, valid_dev = fg.to(device), valid.to(device)
        if not (fg_dev & valid_dev).any() or not (valid_dev & ~fg_dev).any():
            excluded.append(names[i])
            continue

        out_mask = high_norm_tokens(in_norm[None], ratio=3.0)[0]
        bg_mask = valid_dev & (~fg_dev)
        subsets = [
            ("all", valid_dev),
            ("foreground", fg_dev & valid_dev),
            ("background", bg_mask),
            ("high_norm_all", out_mask & valid_dev),
            ("high_norm_bg", out_mask & bg_mask),
            ("non_outlier_bg", (~out_mask) & bg_mask),
        ]
        for sname, smask in subsets:
            st = compute_subset_stats(sname, smask, P, Q, Q_sq, fg_dev, euclid_dist)
            if st is not None:
                cv_rows.append({"model": model_key, "index": i, "image": names[i], **st})

        cached_items.append((i, names[i], s_j.cpu(), c3_j.cpu(), P.cpu(), Q.cpu(), Q_sq.cpu(), fg, valid))
        if (i + 1) % 16 == 0:
            print(f"[feature_origin] {model_key}: {i+1}/{len(dataset)}", flush=True)

    mean_msd_q = float(np.mean(msd_q_list))
    sigma_q = match_sigma(mean_msd_q, grid=GRID)
    blur_q_kernel = gaussian_relocation(sigma_q, grid=GRID)
    blur_probe_kernel = gaussian_relocation(sigmas[12], grid=GRID)

    # Evaluate relocations paired with circularly shifted other_fg for auroc_shuffled
    all_fg = torch.stack([it[7] for it in cached_items])
    all_valid = torch.stack([it[8] for it in cached_items])
    other_all_fg = all_fg.roll(1, 0) if len(cached_items) > 1 else torch.zeros_like(all_fg)

    score_dict = {
        "R0": torch.stack([it[2] for it in cached_items]),
        "C3": torch.stack([it[3] for it in cached_items]),
        "C2_probe": torch.stack([relocate(it[2][None], it[4][None])[0] for it in cached_items]),
        "C2_causal_Q": torch.stack([relocate(it[2][None], it[5][None])[0] for it in cached_items]),
        "C2_causal_Qsq": torch.stack([relocate(it[2][None], it[6][None])[0] for it in cached_items]),
        "blur_probe": torch.stack([relocate(it[2][None], blur_probe_kernel)[0] for it in cached_items]),
        "blur_Q": torch.stack([relocate(it[2][None], blur_q_kernel)[0] for it in cached_items]),
    }

    for m, tensor in score_dict.items():
        res = measures_for_grid(tensor, all_fg, all_valid, other_all_fg, grid=GRID,
                                generator=torch.Generator().manual_seed(SEED))
        for k, it in enumerate(cached_items):
            reloc_rows.append({
                "model": model_key, "index": it[0], "image": it[1], "method": m,
                "enrichment": float(res["enrichment"][k]),
                "auroc": float(res["auroc"][k]),
                "precision": float(res["precision"][k]),
                "mass": float(res["mass"][k]),
            })

    meta = {
        "model": model_key,
        "layer1_self_q_lin_mean": float(np.mean(l1_self_lin)),
        "layer1_self_q_sq_mean": float(np.mean(l1_self_sq)),
        "hook_token_max_diff": hook_err_max,
        "msd_Q": mean_msd_q,
        "sigma_Q": sigma_q,
        "sigma_probe_12": sigmas[12],
        "msd_Qsq": float(np.mean(msd_qsq_list)),
        "excluded": excluded,
    }
    return reloc_rows, cv_rows, meta


def summarize(reloc_rows: list[dict], cv_rows: list[dict], models: list[str], metas: dict) -> dict:
    out = {"meta": metas, "construct_validity": {}, "relocation": {}}
    for mk in models:
        sub_cv = [r for r in cv_rows if r["model"] == mk]
        out["construct_validity"][mk] = {}
        for sname in sorted({r["subset"] for r in sub_cv}):
            srows = [r for r in sub_cv if r["subset"] == sname]
            cols = ["P_self", "Q_self", "Qsq_self", "P_euclid_disp", "Q_euclid_disp", "Qsq_euclid_disp",
                    "JSD_P_Q", "P_fg_mass", "Q_fg_mass", "Qsq_fg_mass", "P_argmax_fg_rate", "Q_argmax_fg_rate"]
            out["construct_validity"][mk][sname] = {
                "n_images": len(srows),
                "mean_tokens_per_image": float(np.mean([r["count"] for r in srows])),
                **{c: bootstrap_mean_ci(np.array([r[c] for r in srows])) for c in cols},
            }

        sub_r = [r for r in reloc_rows if r["model"] == mk]
        methods = sorted({r["method"] for r in sub_r})
        by_m = {m: {r["index"]: r for r in sub_r if r["method"] == m} for m in methods}
        idx = sorted(set.intersection(*(set(v) for v in by_m.values())))
        out["relocation"][mk] = {"n": len(idx)}
        for metric in ("enrichment", "auroc"):
            vecs = {m: np.array([by_m[m][i][metric] for i in idx]) for m in methods}
            out["relocation"][mk][metric] = {
                "means": {m: bootstrap_mean_ci(vecs[m]) for m in methods},
                "contrasts": {
                    "C2_causal_Q_minus_R0": bootstrap_paired_ci(vecs["C2_causal_Q"], vecs["R0"]),
                    "C2_causal_Q_minus_blur_Q": bootstrap_paired_ci(vecs["C2_causal_Q"], vecs["blur_Q"]),
                    "C2_causal_Qsq_minus_R0": bootstrap_paired_ci(vecs["C2_causal_Qsq"], vecs["R0"]),
                    "C2_causal_Q_minus_C2_probe": bootstrap_paired_ci(vecs["C2_causal_Q"], vecs["C2_probe"]),
                    "C2_probe_minus_R0": bootstrap_paired_ci(vecs["C2_probe"], vecs["R0"]),
                    "C3_minus_R0": bootstrap_paired_ci(vecs["C3"], vecs["R0"]),
                    "C3_minus_C2_causal_Q": bootstrap_paired_ci(vecs["C3"], vecs["C2_causal_Q"]),
                },
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--models", nargs="+", default=["deit", "deit_b"])
    parser.add_argument("--occ-batch", type=int, default=64)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.perf_counter()
    data_root = resolve_data_root(args.data_root)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    reloc_all, cv_all, metas = [], [], {}
    for mk in args.models:
        print(f"[feature_origin] Running {mk} on {args.device}...", flush=True)
        r, c, m = evaluate_model(mk, data_root, args.device, args.occ_batch, args.smoke)
        reloc_all.extend(r)
        cv_all.extend(c)
        metas[mk] = m

    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(reloc_all[0].keys()))
        w.writeheader()
        w.writerows(reloc_all)

    with (out / "construct_validity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cv_all[0].keys()))
        w.writeheader()
        w.writerows(cv_all)

    summary = summarize(reloc_all, cv_all, args.models, metas)
    summary["protocol_sha256"] = sha256_file(Path(__file__).resolve().parent / "protocol.md")
    summary["smoke"] = args.smoke
    summary["elapsed_seconds"] = time.perf_counter() - t0
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({mk: summary["relocation"][mk]["enrichment"]["contrasts"] for mk in args.models}, indent=2))


if __name__ == "__main__":
    main()
