# GPU_REML

GPU_REML is a GPU-accelerated statistical framework for SNP heritability
estimation, genetic-variance decomposition, and downstream mixed-model
inference at biobank scale.

The central statistical problem is restricted maximum likelihood (REML)
estimation in linear mixed models where the genetic covariance is defined by one
or more genomic relationship matrices (GRMs). These models are the standard
language for estimating SNP heritability and asking how heritable signal is
distributed across chromosomes, annotations, MAF bins, LD environments, or
user-defined genomic regions.

In the standard formulation, GPU_REML fits a linear mixed model

$$
\begin{aligned}
y &= X\beta + u_1 + \cdots + u_G + e, \\
u_g &\sim \mathcal{N}(0, \sigma_g^2 K_g), \\
e &\sim \mathcal{N}(0, \sigma_e^2 I), \\
V(\theta) &= \sum_g \theta_g K_g + \theta_e I.
\end{aligned}
$$

The restricted log likelihood is

$$\ell_R(\theta)=-\frac{1}{2}\left[\log|V(\theta)|+\log|X^TV(\theta)^{-1}X|+y^TP(\theta)y\right]$$

where

$$P(\theta)=V(\theta)^{-1}-V(\theta)^{-1}X\left(X^TV(\theta)^{-1}X\right)^{-1}X^TV(\theta)^{-1}.$$

Each `K_g` is a genotype-defined covariance component. Let
`a_g = tr(K_g) / n` and `b_j = tr(R_j) / n` denote average diagonal atoms. SNP
heritability is estimated from sample-average variance contributions:

$$
h^2 =
\frac{\sum_g \theta_g a_g}
{\sum_g \theta_g a_g + \sum_j \eta_j b_j}
$$

For unit-trace kernels and one identity residual this reduces to the familiar
ratio of raw variance-component sums. The trace-weighted form is required for
admixed components and effective-rank SMILE normalization.

The computational obstacle is that the natural GRM representation is dense:
constructing, storing, and repeatedly factorizing `n x n` kernels becomes the
bottleneck as cohorts, marker counts, and component counts increase. GPU_REML
therefore keeps the statistical REML model but changes how each `K_g` is applied
numerically. Instead of materializing a GRM, each covariance component is
represented as a matrix-free genotype operator:

$$
K_g v = \frac{Z_g (Z_g^T v)}{m_{\mathrm{eff},g}}
$$

Genotype blocks are decoded on the host, streamed to the GPU, and multiplied in
batches. Evaluating and optimizing the REML likelihood without explicit GRMs
leads to the main numerical machinery in GPU_REML: block PCG solves, Hutchinson
trace estimates, SLQ log-determinant estimates, constrained AI/Fisher updates,
and projected-core preconditioning.

The goal is not only to produce one whole-genome heritability number. GPU_REML is
designed as a method-development workbench for comparing **single-GRM and
multi-GRM covariance representations**. It also includes a SMILE-inspired
weighted-GRM extension, with explicit attribution to the original
[JianqiaoWang/SMILE](https://github.com/JianqiaoWang/SMILE) project. This path
adapts the SMILE idea of introducing a SNP-space weight matrix `W` into the
genetic covariance, while implementing the form that matches GPU_REML's
matrix-free REML engine: a block-diagonal `W`, evaluated without materializing
the sample-space kernel:

$$K_g=\frac{X_gW_gX_g^T}{c_g},\quad W_g=\mathrm{blockdiag}(W_{g,1},\ldots,W_{g,B}),\quad c_g=\frac{\mathrm{tr}(X_gW_gX_g^T)}{n}$$

Each `W_{g,i}` must be a finite symmetric positive-semidefinite dense block.
GPU_REML treats this as a trusted-input contract and does not run a cubic-time
PSD check. Blocks inside one GRM are summed into one variance component;
multiple GRM groups can be supplied when a multi-component REML model is
desired.

The sparse fixed-effect path uses the fitted covariance `V(theta)` to define a
penalized GLS likelihood over candidate SNP effects:

$$
(\hat\alpha_\lambda,\hat b_\lambda)=\arg\min_{\alpha,b}\frac{1}{2}(y-C\alpha-Z_Sb)^TV(\theta)^{-1}(y-C\alpha-Z_Sb)+\lambda\|b\|_1.
$$

The sparse command first alternates this weighted-LASSO step with REML for the
residual `y-C alpha-Z b` in the \(n-\operatorname{rank}(C)\) dimensional space
orthogonal to the complete nuisance design. Supplying the full `C` matrix to
the REML routine profiles its unpenalized coefficients at every candidate
covariance. Since \(P_C C=0\), using a residual that already subtracts the
current nuisance score is algebraically equivalent to applying \(P_C\) to
`y-Z b`, while retaining the numerically convenient residual scale.
Within each candidate problem, coordinate descent is accepted solely when the
active and inactive score-KKT conditions pass at the configured numerical
tolerance; coefficient change is only an active-set scheduling heuristic.
Every evaluated lambda on the complete path must pass that finite-tolerance
certificate. An unsolved path point is reported as a numerical failure rather
than being skipped in favor of the lambda-max empty model. At each covariance
update, held-out validation squared correlation selects lambda before the
variance-component update, and the full-marker KKT scan adds any omitted
violating variants to the candidate set. The KKT tolerance is matched to the
ordinary PCG precision; an independent finite-PCG score on candidate
coordinates is diagnostic rather than a second rejection gate.
The sparse pipeline standardizes the phenotype once at entry and uses that
single analysis scale throughout every LASSO, REML, CHIVE, and prediction step.
The outer loop stops when both the complete fitted mean and primary COHERIT
heritability stabilize, or after the configured maximum number of updates. It
then performs exactly one final validation-selected LASSO update at the
returned covariance, without another
variance update.  Reaching the outer limit is reported as a warning and does
not invalidate a finite final pair.  If that final update itself is unavailable,
the most recent complete covariance-aligned pair is returned with a warning.
The selected lambda/lambda-max ratio is then frozen before the combined
train+validation refit; the held-out test phenotype is never used for lambda
selection. Because that ratio is already fixed, each final-refit LASSO block
solves only the lambda-max warm-start point and the exact target point, rather
than recomputing the unused validation grid. The run stops with this
covariance-aligned LASSO pair and reports
the COHERIT estimator.

The sparse runner supports one fixed covariance model per run. With no
`--component-spec` it uses one whole-genome GRM; with a component spec it uses
that exhaustive, mutually exclusive single-source partition (for example LD2
or LD4). It does not perform Adaptive-K or covariance-tree search. In either
case held-out validation R² selects lambda inside every alpha/theta outer
iteration. The existing `gpu-reml-sparse-validation` orchestrator is the
single-GRM convenience workflow; lower-level partitioned runs pass the same
component spec to both stages. The workflow then freezes the selected
lambda/lambda-max ratio and automatically warm-starts a train+validation final
refit before evaluating the held-out test samples. The test phenotype is never
used for lambda selection.

Sparse-run output semantics are deliberately explicit:

- `estimator_mode` is `coherit` and `computed_estimators` contains only
  `h2_chive`.
- New sparse runs use output schema version 7. Raw/standardized duplicate
  estimator fields and downstream phenotype-scale conversions no longer exist.
- `var_components_lasso_ml` contains one variance contribution per fixed GRM
  followed by the residual variance. Standardized component GRMs use the
  unit-mean-diagonal sparse-COHERIT contract.
- `h2` is the primary total estimate. For a valid covariance-aligned LASSO
  pair it combines the calibrated LASSO quadratic with the
  `var_components_lasso_ml` background and residual components.
- `h2_chive_guarded` is the validated counterpart of `h2_chive`
  and equals the top-level `h2` field whenever the COHERIT branch is valid.
- `q_chive_components` retains the squared fitted-mean and residual-correction terms
  that together form the COHERIT sparse
  variance contribution; the squared fitted-mean term is not exposed as a
  standalone heritability estimator.
- Sparse prediction emits only the `lasso_*` branch: the Lasso
  fixed-SNP score plus its matched background BLUP.
- Every sparse quadratic and variance component is already on the one analysis
  scale established by input phenotype standardization. The summary retains
  only the input normalization metadata needed to transform external outcomes.
- `lasso_branch_valid` validates the COHERIT output. An invalid branch has
  JSON `null` in its guarded fields.
- Ordinary REML is a separate baseline and is never substituted into any
  sparse estimator.
- Sparse prediction follows the same contract. A valid run emits only
  `lasso_*` scores, with no baseline substitution.
- In REML history, `accepted` refers to the current line-search candidate.
  A terminal `ll_down` rejects that candidate and returns the most recent
  accepted variance vector; an intermediate BCD variance block records this
  as a no-update and continues. `converged` is true both when an accepted
  relative-likelihood increment meets the threshold and when every
  backtracking candidate decreases the likelihood (`ll_down`).

## Research Use Cases

GPU_REML is most useful when the scientific question requires more than a
single whole-genome GRM. It is designed to make multi-GRM REML practical by
keeping each covariance component as a streamed genotype operator rather than a
stored dense matrix. This is the main advantage of the project: users can expand
from one GRM to many GRMs while keeping wall time low through GPU batched
products and keeping CPU memory controlled by avoiding explicit `n x n` GRM
storage.

Typical use cases include:

- comparing single-GRM and multi-GRM heritability estimates on the same cohort;
- decomposing SNP heritability across chromosomes, LD environments, MAF bins,
  annotations, or custom SNP sets;
- fitting many covariance components without constructing and storing one dense
  GRM per component;
- benchmarking alternative covariance representations under matched phenotype,
  covariate, and sample filters;
- testing SMILE-style block-diagonal weighted GRMs where dense `W_i` blocks
  encode local SNP covariance or effect-correlation structure.

## Installation

GPU_REML requires Python 3.10 or newer. For large runs, install a GPU-enabled
JAX build before installing GPU_REML.

```bash
git clone https://github.com/Asiandier/GPU_REML.git
cd GPU_REML
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install JAX for the local CUDA driver following the official
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html). For
current NVIDIA CUDA pip wheels, the command is typically:

```bash
python -m pip install -U "jax[cuda13]"
```

Then install GPU_REML:

```bash
python -m pip install -e .
```

Optional PGEN support:

```bash
python -m pip install -e ".[pgen]"
```

Development install:

```bash
python -m pip install -e .
```

Check that JAX can see the GPU:

```bash
python - <<'PY'
import jax
print(jax.devices())
PY
```

CPU-only JAX is sufficient for small examples. Large REML jobs are
intended for GPU execution.

## Quick Start

Single-GRM REML from PLINK1 BED:

```bash
gpu-reml \
  --bed-prefix /path/to/data \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/reml
```

PGEN input:

```bash
gpu-reml \
  --pgen-prefix /path/to/data \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/reml
```

Multiple GRMs from multiple BED prefixes:

```bash
gpu-reml \
  --bed-prefix /path/to/grm1,/path/to/grm2,/path/to/grm3 \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/multi_grm
```

Arbitrary SNP components from one genotype file:

```bash
gpu-reml \
  --bed-prefix /path/to/data \
  --component-spec components.json \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/partitioned
```

Z-score one-shot weak-component merge:

```bash
gpu-reml \
  --merge \
  --bed-prefix /path/to/data \
  --component-spec fine_components.json \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --keep-path keep.txt \
  --out-prefix out/zscore_merge
```

This mode first fits the fine component model, computes component-level
Wald-style z-scores from the fitted variance components and AI matrix, keeps
components with `z >= 1.6448536269514722`, merges all weaker components into one
background GRM, and refits once.

SMILE-style block-diagonal weighted GRM:

```bash
gpu-reml \
  --smile \
  --bed-prefix /path/to/data \
  --w-files W_block_1.npy,W_block_2.npy,W_block_3.npy \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/smile
```

Multiple weighted GRMs are supplied as semicolon-separated groups. Blocks within
a group are summed into one GRM, and each group receives its own variance
component:

```bash
gpu-reml \
  --smile \
  --bed-prefix /path/to/data \
  --grm-groups 'A_1.npy,A_2.npy;B_1.npy,B_2.npy' \
  --pheno-txt pheno.txt \
  --covar-txt covar.txt \
  --out-prefix out/smile_multi
```

Single-GRM sparse COHERIT with validation-R² lambda selection and an automatic
train+validation final refit:

```bash
gpu-reml-sparse-validation \
  --case-id trait_1 \
  --bed-prefix /path/to/data \
  --train-pheno-txt train.pheno \
  --fit-pheno-txt train_plus_validation.pheno \
  --validation-pheno-txt validation.pheno \
  --test-pheno-txt test.pheno \
  --covar-txt covar.txt \
  --train-keep train.keep \
  --validation-keep validation.keep \
  --fit-keep train_plus_validation.keep \
  --test-keep test.keep \
  --out-dir out/sparse_single
```

For a fixed multi-GRM sparse run, add the same exhaustive component spec to
the selection and frozen-ratio refit invocations of
`run_sparse_reml_pipeline.py`:

```bash
python run_sparse_reml_pipeline.py \
  --bed-prefix /path/to/data \
  --component-spec components_ld4.npz \
  --pheno-txt train.pheno \
  --covar-txt covar.txt \
  --keep-path train.keep \
  --prediction-bed-prefix /path/to/data \
  --prediction-covar-txt covar.txt \
  --prediction-keep-path validation.keep \
  --sparsity-validation-pheno-txt validation.pheno \
  --sparsity-validation-out out/ld4.validation.json \
  --lasso-warm-state-out out/ld4.warm.npz \
  --out-prefix out/ld4.selection
```

The final-refit invocation uses the same `--component-spec`, the selected
`--lasso-fixed-lam-ratio`, the G+1 `--variance-components-init`, and the
emitted `--lasso-warm-state-in`.

Continuous-trait marginal GWAS:

```bash
gpu-reml-gwas \
  --bed-prefix /path/to/data \
  --pheno-txt pheno.txt \
  --out-prefix out/gwas
```

Add `--covar-txt covar.txt` when covariates should be included.

The repository-local `run_gpu.sh` launcher remains available for
environment-heavy benchmark runs.

## Component Specifications

Component specs define how SNPs are grouped into GRM components. A JSON spec can
name components and carry metadata:

```json
{
  "components": [
    {
      "name": "maf_0_01",
      "variant_indices": [0, 4, 9],
      "annotation": {"maf_bin": "0-1%"}
    },
    {
      "name": "maf_01_05",
      "variant_indices": [1, 2, 8],
      "annotation": {"maf_bin": "1-5%"}
    }
  ]
}
```

NPZ specs are also supported for compact programmatic construction. See
[docs/component_specs.md](docs/component_specs.md).

## Python API

```python
import jax.numpy as jnp
from GPU_REML import FitConfig, InfinitesimalREMLFitter

cfg = FitConfig(
    bed_prefix="/path/to/data",
    n_rand_vec=100,
    minq_iter=10,
    slq_samples=4,
    slq_m=8,
    precond_rank=500,
    verbose=True,
)

fitter = InfinitesimalREMLFitter(cfg)
result = fitter.fit_infinitesimal(
    y=jnp.asarray(y),
    covar=jnp.asarray(covar),
)
print(result.var_components)
```

Lower-level users can call `fit_reml` with custom `K @ V` operators and diagonal
atoms. This makes it possible to prototype new covariance representations
without rewriting the REML optimizer. The supplied fixed-effect matrix is used
exactly as given, so low-level callers should include an intercept when their
model requires one and remove linearly dependent columns.

## How It Works

At each REML step, GPU_REML needs repeated applications of:

$$
H(\theta)V = \theta_e V + \sum_g \theta_g K_g V
$$

The implementation builds this product from streamed genotype blocks. REML
evaluation then combines:

- block PCG solves for `H^-1 [X | y | random probes]`;
- Hutchinson probes for trace terms in the score;
- stochastic Lanczos quadrature for `log|H|`;
- one-pass affine Lanczos reuse for the single-GRM identity-residual model;
- projected Fisher / AI-style variance-component updates with nonnegative
  genetic-variance constraints;
- a projected-core preconditioner `dI + U C(theta) U.T` that captures leading
  covariance structure. Residual SLQ keeps a fit-wide fixed reference, while
  PCG independently rebuilds its basis after accepted nonterminal updates.

For SMILE-style weighted kernels, the same REML loop is reused after replacing
the standard GRM operator by the block-diagonal weighted operator. This keeps the
new covariance representation isolated from the ordinary single-GRM, multi-GRM,
partitioned, and sparse paths.

For routine runs, the most important user-facing resource controls are the GPU
budget and the genotype-streaming ring depth.

See [the mathematical overview](docs/mathematical_overview.md) for the score,
AI, trace, SLQ, preconditioner, and heritability formulas, and
[the architecture guide](docs/architecture.md) for module boundaries and the
fit lifecycle.

## Key Runtime Parameters

- `--gpu-budget-gib`: planner-side budget for **active** GPU allocations. The
  planner uses it to choose the streamed SNP call width and the size of
  GPU-resident work arrays, including REML random-probe blocks and
  projected-core state. If omitted or set to `0`, GPU_REML uses 85% of the
  currently available GPU memory estimate. This is not a hard `nvidia-smi`
  process-memory cap: JAX's pooled allocator can retain inactive blocks for
  reuse, and CUDA also owns context/workspace memory. GPU_REML reports the JAX
  `peak_active` value and, when the backend exposes it, `peak_reserved` at the
  end of every CLI run so users can distinguish real live use from allocator
  retention. Lower this budget to leave more VRAM for other processes. For memory diagnostics,
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform` releases allocations eagerly, at a
  potential performance cost.
- `--ring-depth`: number of CPU-side staging buffers used for genotype
  streaming. This is the main knob for controlling CPU memory peak during data
  movement. Larger values can give smoother host-to-GPU streaming but allocate
  more pinned/staging memory on the CPU. The default `0` lets the planner choose
  a conservative value.
- `GPU_REML_MATMUL_PRECISION`: JAX matmul precision policy. The robust default
  is `highest`; an alternative should be used only after validating numerical
  agreement on the target GPU.
- `--pcg-tol`: ordinary PCG tolerance used by screening, candidate construction,
  and outer iterations; the default is `5e-3`.
- `--kkt-tol` and `--kkt-rel-tol`: absolute and lambda-scaled tolerances for
  the signed score-KKT conditions.  Their effective minimum is
  `max(1e-4, 2*pcg_tol)`, so the certificate does not demand more precision
  than its PCG inputs provide.
- `--outer-max`: maximum number of variance updates; the default is `20`.
- `--sparsity-validation-pheno-txt` and `--sparsity-validation-out`: required
  together for model selection. The complete lambda path is evaluated on the
  validation samples inside every alpha/theta outer iteration.
- `--effect-rel-tol`: relative tolerance for change in the complete fitted
  fixed mean; the default is `5e-2`.

The startup report labels the pinned streaming ring as `Host memory plan (CPU
RAM, not GPU VRAM)`. A line such as `streaming_ring=35.4GiB` therefore describes
host RAM. At shutdown, `peak_active` is the value to compare with the planner's
active estimate; `peak_reserved` is closer to what `nvidia-smi` observes.

## Validation

Build a wheel:

```bash
python -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir /tmp/gpu_reml_wheel /path/to/GPU_REML
```

## Limitations

- REML likelihood terms use randomized approximations; results can vary with
  seed and SLQ/Hutchinson settings.
- A PCG residual tolerance does not itself give an analytic coordinatewise
  bound on KKT-score error.  Reported KKT status is a finite-tolerance
  numerical certificate, not a proof of exact optimality.
- The package currently focuses on continuous traits.
- GPU performance depends on JAX/CUDA versions, PCIe bandwidth, call width,
  sample size, SNP count, and component count.
- This is research software. Validate settings against small exact references or
  matched external software before using it for production scientific
  conclusions.
- No public license has been selected yet. Do not redistribute until the project
  owner adds an explicit license.
