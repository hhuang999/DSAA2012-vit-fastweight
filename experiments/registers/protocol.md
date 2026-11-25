# Cross-Paradigm & Register-Trained Generalization Protocol (Critique 3.1)

Fixed before execution on branch `paper/v22`.

## Purpose
Test whether the attention--contribution decoupling mechanism and value-weighting gains:
1. Generalise beyond supervised ImageNet DeiTs to self-supervised DINOv2 (`vit_small_patch14_dinov2.lvd142m`, `vit_base_patch14_dinov2.lvd142m`) and multimodal contrastive CLIP (`vit_base_patch16_clip_224.openai`), which also exhibit high-input-norm background artifacts; and
2. Collapse as predicted by our mechanism in matched register-trained controls (`vit_small_patch14_reg4_dinov2.lvd142m`, `vit_base_patch14_reg4_dinov2.lvd142m`) trained with the exact same objective and LVD-142M data plus 4 register tokens.

## Cohort, Resolution & Layers
- **Dataset**: Oxford-IIIT Pet official test trimaps, exact 384 names from `experiments/spatial/selected_names.txt`, preprocessed with each model's native `timm` evaluation transform (`518x518`, `37x37` grid for DINOv2 variants; `224x224`, `14x14` grid for CLIP).
- **Layers**: Last 4 blocks (`layers = (9, 10, 11, 12)`).
- **Prefix handling**: `prefix = model.num_prefix_tokens` (`1` for CLS-only models, `5` for `reg4` models). Queries sum over all `prefix + N` tokens; scores are evaluated on the `N` spatial patches.
- **LayerScale check**: DINOv2 blocks apply residual LayerScale (`ls1`). Log both standard `C3` (`||W_O^h v_j^h||_2`) and `C3_ls1` (`||ls1(W_O^h v_j^h)||_2`) as an explicit check.

## Readouts & Inference
- Readouts: `R0`, `C3`, `Z_out_3.0`, `Soft_out_only`, `NonOut_vw_zeroOut`, `NonOut_ring_shuffle` (5 seeded within-ring permutations averaged per image), plus `C3_ls1` where `ls1.gamma` exists.
- Per-layer diagnostics: fraction of patches with block-input norm $>3\times$ image median, attention share on those patches, and ratio of their mean projected value norm to the non-outlier median value norm.
- Metrics & bootstrap: Same-ring enrichment (generalised cleanly to `grid = 37` or `14`) and foreground AUROC; 1,000 paired bootstrap resamples, seed `0`, 95% percentile CIs.
