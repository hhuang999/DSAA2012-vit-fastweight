# Orthogonal Decomposition & Threshold Sweep Protocol (Critique 3.3)

Fixed before execution on branch `paper/v22`.

## Purpose
Decompose the alignment gap between binary `high-norm zeroing` ($Z_{\text{out}}(3.0)$) and continuous projected `value weighting` ($C_3$) into four orthogonal components:
1. Soft vs. binary handling of high-input-norm outliers (`Soft_out_only` vs. `Z_out_3.0`);
2. Continuous value-norm reweighting strictly among non-outlier patches (`NonOut_vw_zeroOut` vs. `Z_out_3.0`);
3. Retaining outlier patches with soft value norms on top of non-outlier weighting (`C3` vs. `NonOut_vw_zeroOut`);
4. Equal-radius spatial norm permutation restricted strictly to non-outlier patches while holding outlier norms fixed (`C3` vs. `NonOut_ring_shuffle`).

In addition, test a threshold sweep $\tau \in \{1.5, 2.0, 2.5, 3.0, 4.0\}$ on binary zeroing ($Z_{\text{out}}(\tau)$) to check whether relaxed binary outlier thresholds recover the full continuous $C_3$ gain.

## Cohort, Models & Preprocessing
- **Dataset**: Oxford-IIIT Pet official test trimaps, exact 384 images in `experiments/spatial/selected_names.txt`. Crops lacking either valid foreground or valid background patches are excluded consistently per model.
- **Models**: Frozen `deit` (DeiT-S/16), `deit_b` (DeiT-B/16), and `dino` (DINO ViT-S/16 backbone, using its native 256->224 crop 0.875 transform).
- **Layers**: Layers 9--12 summed.

## Definitions (per layer $l$, head $h$, patch $j$)
Let $a_{hj} = \sum_i a_{hij}$, $z_{hj} = \|W_O^h v_{hj}\|_2$, and $O_j = \mathbb{I}[\|u_j\|_2 > \tau \cdot \mathrm{median}_k(\|u_k\|_2)]$ ($\tau = 3.0$ unless otherwise stated). Let $m_h = \mathrm{median}_{j \notin O}(z_{hj})$ (or $\mathrm{median}_j(z_{hj})$ if all patches are in $O$).
- `R0`: $\sum_h a_{hj}$
- `C3`: $\sum_h a_{hj} z_{hj}$
- `Z_out_3.0` (and `Z_out_{tau}` for $\tau \in \{1.5, 2.0, 2.5, 4.0\}$): $\sum_h a_{hj} \mathbb{I}[\neg O_j]$
- `Soft_out_only`: $\sum_h a_{hj} w_{hj}$, where $w_{hj} = z_{hj}$ if $O_j$ else $m_h$
- `NonOut_vw_zeroOut`: $\sum_h a_{hj} z_{hj} \mathbb{I}[\neg O_j]$
- `NonOut_vw_keepOut`: $\sum_h a_{hj} w_{hj}$, where $w_{hj} = m_h$ if $O_j$ else $z_{hj}$
- `NonOut_ring_shuffle`: for each of 5 deterministic seeds, keep $z_{hj}$ fixed on $O$, permute non-outlier $z_{hj}$ uniformly within each equal-radius ring among non-outliers in that ring; evaluate enrichment and AUROC per permutation and average the 5 scores per image (eliminates single-draw permutation noise).

## Metrics & Statistical Inference
- Primary metrics: `enrichment` (same-ring foreground enrichment) and `auroc` (patch-level foreground AUROC).
- Paired bootstrap over usable images: 1,000 resamples, RNG seed `0`, 95% percentile intervals.
