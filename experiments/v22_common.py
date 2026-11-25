"""Shared utilities for v22 exploratory studies (orthogonal decomposition, registers, feature origin, CLS deletion)."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, InterpolationMode, Resize
from torchvision.transforms import functional as TF

from fastweight_vit.data.masks import token_foreground
from fastweight_vit.enrichment import measures as fast_measures_14
from fastweight_vit.execution.execute import split_qkv
from fastweight_vit.readings import output_gram, received_and_length
from fastweight_vit.tokens import centre_rings

ROOT = Path(__file__).resolve().parents[1]
SPATIAL_NAMES = ROOT / "experiments/spatial/selected_names.txt"
CLS_NAMES = ROOT / "experiments/cls/selected_names.txt"
SEED = 2026


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spatial_names(n: int | None = None) -> list[str]:
    names = [line.strip() for line in SPATIAL_NAMES.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != 384 or len(set(names)) != 384:
        raise ValueError("Expected 384 unique names in experiments/spatial/selected_names.txt")
    return names if n is None else names[:n]


def load_cls_names(n: int | None = None) -> list[str]:
    names = [line.strip() for line in CLS_NAMES.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != 128 or len(set(names)) != 128:
        raise ValueError("Expected 128 unique names in experiments/cls/selected_names.txt")
    return names if n is None else names[:n]


class OxfordPetsSubset(Dataset):
    """Oxford-IIIT Pet subset aligned to any model's eval preprocessing and patch size."""

    def __init__(self, names: list[str], data_root: Path, model_or_ref, patch_size: int = 16, preprocessing=None):
        self.names = list(names)
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.preprocessing = preprocessing or resolve_model_data_config(model_or_ref)
        self.transform = create_transform(**self.preprocessing, is_training=False)
        self.resize, self.crop = self.transform.transforms[:2]
        if not isinstance(self.resize, Resize) or not isinstance(self.crop, CenterCrop):
            raise ValueError(f"Expected Resize+CenterCrop first, got {self.transform.transforms[:2]}")
        crop_hw = self.crop.size
        crop_size = crop_hw[0] if isinstance(crop_hw, (tuple, list)) else int(crop_hw)
        if crop_size % patch_size != 0:
            raise ValueError(f"Crop size {crop_size} is not divisible by patch_size {patch_size}")
        self.grid = crop_size // patch_size
        missing = [
            name for name in self.names
            if not (self.data_root / "images" / f"{name}.jpg").is_file()
            or not (self.data_root / "annotations/trimaps" / f"{name}.png").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} Oxford-IIIT Pet files, e.g. {missing[:3]}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.data_root / "images" / f"{name}.jpg") as img:
            image = img.convert("RGB")
        with Image.open(self.data_root / "annotations/trimaps" / f"{name}.png") as tri:
            trimap = tri.convert("L")
        if image.size != trimap.size:
            raise ValueError(f"Image and trimap dimensions differ for {name}")
        mask = TF.resize(trimap, self.resize.size, interpolation=InterpolationMode.NEAREST,
                         max_size=self.resize.max_size)
        mask = torch.from_numpy(np.array(TF.center_crop(mask, self.crop.size), dtype=np.int64))
        foreground, valid = token_foreground(mask, self.patch_size)
        return self.transform(image), foreground, valid, index


def auroc_per_image(scores: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    out = torch.full((scores.shape[0],), float("nan"), dtype=torch.float64)
    for n in range(scores.shape[0]):
        s = scores[n][valid[n]].double()
        p = positive[n][valid[n]]
        pos, neg = s[p], s[~p]
        if len(pos) and len(neg):
            greater = (pos[:, None] > neg[None, :]).double().mean()
            ties = (pos[:, None] == neg[None, :]).double().mean()
            out[n] = greater + 0.5 * ties
    return out


def measures_for_grid(scores: torch.Tensor, foreground: torch.Tensor, valid: torch.Tensor,
                      other_foreground: torch.Tensor, grid: int = 14,
                      generator: torch.Generator | None = None) -> dict[str, np.ndarray]:
    """Grid-generalized enrichment/AUROC/precision/mass matching fastweight_vit.enrichment.measures when grid==14."""
    scores_f, fg_c, valid_c = scores.detach().float().cpu(), foreground.cpu(), valid.cpu()
    other_c = other_foreground.cpu()
    fg = fg_c & valid_c
    positive = scores_f.clamp_min(0) * valid_c
    mass = (positive * fg).sum(1) / positive.sum(1)
    peak = scores_f.masked_fill(~valid_c, -torch.inf).argmax(1)
    pointing = fg.gather(1, peak[:, None])[:, 0].float()
    n_fg = fg.sum(1)
    noise = torch.rand(scores_f.shape, generator=generator) * 1e-6 * (scores_f.abs().amax(1, keepdim=True) + 1e-12)
    order = (scores_f + noise).masked_fill(~valid_c, -torch.inf).argsort(1, descending=True)
    selected = torch.zeros_like(fg)
    for n in range(scores_f.shape[0]):
        selected[n, order[n, :int(n_fg[n])]] = True
    precision = (selected & fg).sum(1) / n_fg.clamp_min(1)
    rings = centre_rings(grid, prefix=1)[1:]
    unique_rings = rings.unique().tolist()
    enrichment = torch.zeros(scores_f.shape[0], dtype=torch.float64)
    for n in range(scores_f.shape[0]):
        weights, total = 0, 0.0
        for ring in unique_rings:
            members = (rings == ring) & valid_c[n]
            chosen = members & selected[n]
            if chosen.any():
                rate = fg[n][chosen].double().mean() - fg[n][members].double().mean()
                total += float(rate) * int(chosen.sum())
                weights += int(chosen.sum())
        enrichment[n] = total / weights if weights else float("nan")
    usable = (n_fg > 0) & ((valid_c & ~fg).sum(1) > 0)
    nan = torch.tensor(float("nan"))

    rows, cols = np.divmod(np.arange(grid * grid), grid)
    fg_np = fg.numpy()
    area = fg_np.sum(1) / np.maximum(valid_c.numpy().sum(1), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (fg_np * rows).sum(1) / fg_np.sum(1)
        c = (fg_np * cols).sum(1) / fg_np.sum(1)
    distance = np.hypot(r - (grid - 1) / 2, c - (grid - 1) / 2)

    out = dict(
        mass=torch.where(usable & (positive.sum(1) > 0), mass, nan).numpy(),
        pointing=torch.where(usable, pointing, nan).numpy(),
        auroc=auroc_per_image(scores_f, fg, valid_c).numpy(),
        auroc_shuffled=auroc_per_image(scores_f, other_c & valid_c, valid_c).numpy(),
        precision=torch.where(usable, precision, nan).numpy(),
        enrichment=enrichment.numpy(),
        object_area=area,
        object_distance=distance,
    )
    if grid == 14 and generator is None:
        # Note: seeded generator comparison is done explicitly by callers.
        pass
    return out


def verify_measures_grid14_equivalence(scores, fg, valid, other_fg, seed=2026):
    g1 = torch.Generator().manual_seed(seed)
    g2 = torch.Generator().manual_seed(seed)
    a = fast_measures_14(scores, fg, valid, other_fg, generator=g1)
    b = measures_for_grid(scores, fg, valid, other_fg, grid=14, generator=g2)
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-6, atol=1e-6, equal_nan=True)


@contextmanager
def capture_per_head(model, layers, prefix=None):
    """Capture per-head received attention, projected value norms, value norms, ls1-scaled norms, and block-input norms."""
    backbone = getattr(model, "backbone", model)
    prefix = backbone.num_prefix_tokens if prefix is None else prefix
    store = {layer: {} for layer in layers}
    saved = {}
    handles = []

    def make_forward(layer, attention, ls1_module):
        gram = output_gram(attention)
        ls1_gamma = getattr(ls1_module, "gamma", None)

        def forward(x, attn_mask=None, is_causal=False):
            if attn_mask is not None or is_causal:
                raise ValueError("Expected unmasked, noncausal self-attention")
            if not isinstance(attention.norm, torch.nn.Identity):
                raise ValueError("Expected Identity attention norm")
            batch, tokens, _ = x.shape
            q, k, v = split_qkv(attention, x)
            p = ((q * attention.scale) @ k.transpose(-2, -1)).softmax(-1)
            received, length = received_and_length(p, v, gram)
            store[layer]["received"] = received[:, :, prefix:]  # [B, H, N]
            store[layer]["length"] = length[:, :, prefix:]      # [B, H, N]
            store[layer]["value_norm"] = v.norm(dim=-1)[:, :, prefix:]
            if ls1_gamma is not None:
                weight = attention.proj.weight.reshape(-1, attention.num_heads, attention.head_dim)
                z = torch.einsum("dhe,bhje->bhjd", weight, v[:, :, prefix:, :])
                store[layer]["length_ls1"] = (z * ls1_gamma[None, None, None, :]).norm(dim=-1)
            out = (p @ v).transpose(1, 2).reshape(batch, tokens, -1)
            return attention.proj_drop(attention.proj(attention.norm(out)))

        return forward

    def norm_hook(layer):
        def hook(module, inputs):
            store[layer]["norms"] = inputs[0][:, prefix:].float().norm(dim=-1)
        return hook

    try:
        for layer in layers:
            block = backbone.blocks[layer - 1]
            saved[layer] = block.attn.__dict__.get("forward")
            block.attn.forward = make_forward(layer, block.attn, getattr(block, "ls1", None))
            handles.append(block.register_forward_pre_hook(norm_hook(layer)))
        yield store
    finally:
        for h in handles:
            h.remove()
        for layer, orig in saved.items():
            attn = backbone.blocks[layer - 1].attn
            if orig is None:
                del attn.forward
            else:
                attn.forward = orig


def bootstrap_paired_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 1000, seed: int = 0) -> dict:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    boot = diff[draws].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"mean": float(diff.mean()), "ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0)}


def bootstrap_mean_ci(a: np.ndarray, n_resamples: int = 1000, seed: int = 0) -> dict:
    vals = np.asarray(a, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(vals), size=(n_resamples, len(vals)))
    boot = vals[draws].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"mean": float(vals.mean()), "ci95": [float(lo), float(hi)]}


def resolve_data_root(cli_root: str | Path | None = None) -> Path:
    if cli_root is not None:
        return Path(cli_root)
    return Path(os.environ.get("OXFORD_PETS_ROOT", ROOT / "data/oxford_pets"))
