"""Classifier-aligned (CLS-only) input-patch deletion evaluation on 128 Oxford-IIIT Pet crops."""

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
    load_cls_names,
    resolve_data_root,
    sha256_file,
)
from fastweight_vit.execution.execute import split_qkv  # noqa: E402
from fastweight_vit.faithfulness import curve, rank  # noqa: E402
from fastweight_vit.models import IMAGENET_PREPROCESSING, load_model  # noqa: E402
from fastweight_vit.readings import capture  # noqa: E402

BUDGETS = (0, 1, 2, 4, 8, 16, 32, 64, 98, 147)
METHODS = ("all_r0", "all_c3", "cls_attention", "cls_sum_head", "cls_net", "all_r0_9_12", "all_c3_9_12")


def layer12_maps(model, images: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Compute layer-12 all-query and CLS-only maps matching experiments/cls/run_cls.py, plus 9-12 refs."""
    with torch.no_grad(), capture(model, (9, 10, 11, 12), per_head=True) as store:
        # Also hook block 12 input to get exact CLS row p[:, :, 0, 1:]
        observed = {}

        def pre_hook(_mod, inputs):
            observed["x"] = model.blocks[11].norm1(inputs[0])

        h = model.blocks[11].register_forward_pre_hook(pre_hook)
        try:
            logits = model(images)
        finally:
            h.remove()

    attn = model.blocks[11].attn
    q, k, v = split_qkv(attn, observed["x"])
    p = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(-1)
    p_cls = p[:, :, 0, 1:]  # [B, H, 196]
    weight = attn.proj.weight.reshape(-1, attn.num_heads, attn.head_dim)
    proj_v = torch.einsum("dhe,bhje->bhjd", weight, v[:, :, 1:, :])
    head_len = store[12]["length"]
    assert torch.allclose(head_len, proj_v.norm(dim=-1), rtol=3e-5, atol=1e-4)

    maps = {
        "all_r0": store[12]["r0"],
        "all_c3": store[12]["c3"],
        "cls_attention": p_cls.sum(dim=1),
        "cls_sum_head": (p_cls * head_len).sum(dim=1),
        "cls_net": (p_cls[:, :, :, None] * proj_v).sum(dim=1).norm(dim=-1),
        "all_r0_9_12": sum(store[l]["r0"] for l in (9, 10, 11, 12)),
        "all_c3_9_12": sum(store[l]["c3"] for l in (9, 10, 11, 12)),
    }
    return maps, logits


def trapz_auc(x_frac: np.ndarray, y_vals: np.ndarray) -> float:
    return float(np.trapezoid(y_vals, x_frac) if hasattr(np, "trapezoid") else np.trapz(y_vals, x_frac))


def evaluate_model(model_key: str, data_root: Path, device: str, batch_size: int, smoke: int):
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(model_key, device=device)
    names = load_cls_names(smoke if smoke else None)
    dataset = OxfordPetsSubset(names, data_root, model, patch_size=16, preprocessing=IMAGENET_PREPROCESSING[model_key])
    x_frac = np.asarray(BUDGETS, dtype=np.float64) / 196.0

    per_img_rows = []
    curve_rows = []

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        idx_list = list(range(start, end))
        images = torch.stack([dataset[i][0] for i in idx_list]).to(device)
        maps, original = layer12_maps(model, images)

        for method in METHODS:
            gen = torch.Generator(device=device).manual_seed(SEED * 1_000_003 + start)
            order = rank(maps[method], gen)
            drop_top = curve(model, images, original, order, BUDGETS, "drop_top", prefix=1)
            drop_bot = curve(model, images, original, order, BUDGETS, "drop_bottom", prefix=1)

            for b in BUDGETS:
                for cond_name, cdict in (("drop_top", drop_top), ("drop_bottom", drop_bot)):
                    for k_i, img_idx in enumerate(idx_list):
                        curve_rows.append({
                            "model": model_key, "index": img_idx, "image": names[img_idx],
                            "method": method, "condition": cond_name, "budget": b,
                            "agree": float(cdict[b]["agree"][k_i]),
                            "p_target": float(cdict[b]["p_target"][k_i]),
                            "kl": float(cdict[b]["kl"][k_i]),
                        })

            for k_i, img_idx in enumerate(idx_list):
                top_agree = np.array([float(drop_top[b]["agree"][k_i]) for b in BUDGETS])
                bot_agree = np.array([float(drop_bot[b]["agree"][k_i]) for b in BUDGETS])
                top_p = np.array([float(drop_top[b]["p_target"][k_i]) for b in BUDGETS])
                bot_p = np.array([float(drop_bot[b]["p_target"][k_i]) for b in BUDGETS])
                top_kl = np.array([float(drop_top[b]["kl"][k_i]) for b in BUDGETS])
                bot_kl = np.array([float(drop_bot[b]["kl"][k_i]) for b in BUDGETS])
                per_img_rows.append({
                    "model": model_key, "index": img_idx, "image": names[img_idx], "method": method,
                    "srg_agree": trapz_auc(x_frac, bot_agree) - trapz_auc(x_frac, top_agree),
                    "prob_sep": trapz_auc(x_frac, bot_p) - trapz_auc(x_frac, top_p),
                    "kl_sep": trapz_auc(x_frac, top_kl) - trapz_auc(x_frac, bot_kl),
                })
        print(f"[cls_deletion] {model_key}: {end}/{len(dataset)}", flush=True)

    return per_img_rows, curve_rows


def summarize_model(rows: list[dict], model_key: str) -> dict:
    sub = [r for r in rows if r["model"] == model_key]
    by_m = {m: {r["index"]: r for r in sub if r["method"] == m} for m in METHODS}
    idx = sorted(set.intersection(*(set(v) for v in by_m.values())))
    out = {"n": len(idx), "endpoints": {}}
    pairs = [
        ("all_c3_minus_all_r0", "all_c3", "all_r0"),
        ("cls_attention_minus_all_r0", "cls_attention", "all_r0"),
        ("cls_sum_head_minus_all_c3", "cls_sum_head", "all_c3"),
        ("cls_sum_head_minus_cls_attention", "cls_sum_head", "cls_attention"),
        ("cls_net_minus_cls_sum_head", "cls_net", "cls_sum_head"),
        ("cls_net_minus_all_c3", "cls_net", "all_c3"),
        ("cls_net_minus_all_r0", "cls_net", "all_r0"),
        ("all_c3_9_12_minus_all_r0_9_12", "all_c3_9_12", "all_r0_9_12"),
    ]
    for ep in ("srg_agree", "prob_sep", "kl_sep"):
        vecs = {m: np.array([by_m[m][i][ep] for i in idx]) for m in METHODS}
        out["endpoints"][ep] = {
            "means": {m: bootstrap_mean_ci(vecs[m]) for m in METHODS},
            "contrasts": {label: bootstrap_paired_ci(vecs[a], vecs[b]) for label, a, b in pairs},
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--models", nargs="+", default=["deit", "deit_b"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.perf_counter()
    data_root = resolve_data_root(args.data_root)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    all_imgs, all_curves = [], []
    for mk in args.models:
        print(f"[cls_deletion] Running {mk} on {args.device}...", flush=True)
        imgs, curves = evaluate_model(mk, data_root, args.device, args.batch_size, args.smoke)
        all_imgs.extend(imgs)
        all_curves.extend(curves)

    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_imgs[0].keys()))
        w.writeheader()
        w.writerows(all_imgs)

    with (out / "budget_curves.csv.gz").open("wb") as raw_f:
        import gzip
        with gzip.GzipFile(fileobj=raw_f, mode="wb", mtime=0) as gz:
            import io
            txt = io.TextIOWrapper(gz, encoding="utf-8", newline="")
            w = csv.DictWriter(txt, fieldnames=list(all_curves[0].keys()))
            w.writeheader()
            w.writerows(all_curves)
            txt.flush()

    summary = {
        "protocol_sha256": sha256_file(Path(__file__).resolve().parent / "protocol.md"),
        "smoke": args.smoke,
        "elapsed_seconds": time.perf_counter() - t0,
        "models": {mk: summarize_model(all_imgs, mk) for mk in args.models},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({mk: summary["models"][mk]["endpoints"]["srg_agree"]["contrasts"] for mk in args.models}, indent=2))


if __name__ == "__main__":
    main()
