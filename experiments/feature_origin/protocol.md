# Causal Feature Origin vs. Position-Probe Protocol (Critique 3.2)

Fixed before execution on branch `paper/v22`.

## Purpose
Resolve the construct-validity gap between linear slot-identity probes $P(p \mid x_j)$ and actual causal feature origin $Q(p \mid x_j)$ in layer 12 of frozen DeiT-S and DeiT-B, and test whether causal-origin relocation cures deep background attention peaks.

## Cohort & Probes
- **Cohort**: First 64 deterministic images from `experiments/cls/selected_names.txt` (saved in `experiments/feature_origin/selected_names.txt`).
- **Models**: Frozen `deit` (DeiT-S) and `deit_b` (DeiT-B), layer 12.
- **Probes**: Exact pre-trained layer-12 linear position probes from `outputs/diagnose_probe/{deit,deit_b}/probes.npz`.

## Causal Feature Origin Measurement
For each image, run the unmodified forward pass to obtain layer-12 normalised attention-input tokens $x^{(12)}_{\mathrm{orig}} \in \mathbb{R}^{196 \times D}$, block-12 input norms $\|u_j\|_2$, and received attention $s_j = \sum_{h,i} a^{(12)}_{hij}$.
Then for each input patch $p \in \{0,\dots,195\}$, replace the $16{\times}16$ pixels of patch $p$ in the normalised image tensor with `0.0` (ImageNet mean), and measure the resulting layer-12 representation shift across all tokens:
\[
\Delta_{j,p} = \|x^{(12)}_{\mathrm{occ}(p), j} - x^{(12)}_{\mathrm{orig}, j}\|_2.
\]
Define two normalised causal feature-origin distributions per token $j$:
- Linear sensitivity: $Q_{j,p} = \Delta_{j,p} / \sum_{p'} \Delta_{j,p'}$
- Squared sensitivity: $Q^{\mathrm{sq}}_{j,p} = \Delta_{j,p}^2 / \sum_{p'} \Delta_{j,p'}^2$

Layer-1 sanity check (first 4 images): verify that at block 1 (`norm1` before mixing), mean self-patch squared causal energy $Q^{\mathrm{sq},(1)}_{j,j} > 0.98$ (with TF32 disabled).

## Construct-Validity Comparisons & Relocations
1. Compare $P$ and $Q$ across all patches, background patches, and high-input-norm outlier patches ($>3\times$ median): self-slot mass ($P_{j,j}$ vs. $Q_{j,j}$), expected displacement (in patches), Jensen--Shannon divergence, and mass placed on foreground patches.
2. Evaluate layer-12 readouts on same-ring enrichment and foreground AUROC: `R0`, `C3`, `C2_probe` ($s P$), `C2_causal_Q` ($s Q$), `C2_causal_Qsq` ($s Q^{\mathrm{sq}}$), and matched-displacement Gaussian blur `blur_Q` whose MSD matches $\mathrm{MSD}(Q)$. Paired bootstrap: 1,000 resamples, seed `0`.
