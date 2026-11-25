# Classifier-Aligned (CLS-only) Input-Patch Deletion Protocol

Fixed before execution on branch `paper/v22`.

## Purpose
Test whether restricting layer-12 value weighting from all 197 queries (`all_c3`) to the classification token (`cls_sum_head`), or keeping cross-head cancellation (`cls_net = ||sum_h a_{CLS,j}^h W_O^h v_j^h||_2`), improves input-patch deletion faithfulness relative to `all_r0` and `all_c3`.

## Cohort & Models
- **Cohort**: Exact 128 Oxford-IIIT Pet crops in `experiments/cls/selected_names.txt`.
- **Models**: Frozen `deit` (DeiT-S) and `deit_b` (DeiT-B), using each model's native evaluation preprocessing and pretrained ImageNet-1k classification head.

## Readouts & Deletion Protocol
Compute per-image layer-12 maps matching `experiments/cls/run_cls.py`: `all_r0`, `all_c3`, `cls_attention`, `cls_sum_head`, `cls_net`, plus 4-layer references `all_r0_9_12` and `all_c3_9_12`.
Run `fastweight_vit.faithfulness.curve` with exact paper removal budgets $\{0, 1, 2, 4, 8, 16, 32, 64, 98, 147\}$ under both `drop_top` (MoRF) and `drop_bottom` (LeRF), tying scores identically with seeded tie-breaking (`SEED = 2026`).
Integrate over $b / 196$ via trapezoidal rule to obtain all three faithfulness endpoints:
1. Binary top-1 retention SRG: $\mathrm{AUC}_{\mathrm{LeRF}}(\text{agree}) - \mathrm{AUC}_{\mathrm{MoRF}}(\text{agree})$;
2. Continuous top-1 probability separation: $\mathrm{AUC}_{\mathrm{LeRF}}(p_{\text{top1}}) - \mathrm{AUC}_{\mathrm{MoRF}}(p_{\text{top1}})$;
3. Continuous KL separation: $\mathrm{AUC}_{\mathrm{MoRF}}(\mathrm{KL}) - \mathrm{AUC}_{\mathrm{LeRF}}(\mathrm{KL})$.

Paired bootstrap contrasts over the 128 images: 1,000 resamples, RNG seed `0`, 95% percentile intervals.
