# Code Architecture

GPU_REML is organized around one contract: a covariance component must expose
`K @ V` and diagonal information without constructing `K`.

## Layer Map

| Layer | Main modules | Responsibility |
| --- | --- | --- |
| CLI and planning | `run_reml_pipeline.py`, `run_sparse_reml_pipeline.py`, `suggest_params_v3.py` | Parse user inputs, estimate memory peaks, choose call width/ring depth, and report results. |
| Data alignment | `data_utils.py`, `geno_source.py`, `component_spec.py`, `admixed_cov.py` | Align samples and variants, validate component definitions, and expose BED/PGEN sources. |
| Streaming | `geno_stream.py`, `sparse_stream.py`, `block_backend.py` | Build packed host caches, retain standardization statistics, prefetch blocks, and manage pinned staging buffers. |
| Matrix-free operators | `kv_impl.py`, `smile_block_w.py` | Decode blocks and implement standard, partitioned, multi-GRM, admixed, weighted, transpose, and prediction products. |
| Numerical solver | `reml.py`, `pcg.py`, `precond.py` | Form the REML projection, Hutchinson scores, SLQ objective, AI matrix, constrained updates, and projected-core preconditioner. |
| Public model API | `reml_model.py` | Build compatible operator bundles, manage preconditioner lifecycle, fit replicates, estimate effects, and predict. |
| Downstream models | `lasso_cd.py`, `gwas.py` | Reuse the fitted covariance for sparse GLS and marginal GWAS. |
| Sparse mean information | `sparse_information.py` | Selected-span log determinant, score correction and paired sparse-variance correction, without sample-square matrices. |
| Adaptive partition | `run_sparse_pipeline.py`, `adaptive_ld.py`, `score_process.py` | Orchestrate the path, stream shared boundary contractions, and calibrate trace-defined quadratic scores. |

## Fit Lifecycle

1. The CLI aligns phenotype, covariates, sample filters, and genotype sources.
2. The planner estimates phase-specific GPU peaks, including the persistent
   `K_i @ Vrand` cache, and chooses a feasible streamed call width.
3. `GenoBlockStreamer` builds or opens the packed host cache and computes the
   training standardization statistics.
4. `InfinitesimalREMLFitter._assemble_reml_operators` returns an
   `_OperatorBundle`: component matvecs, diagonals, a weighted `H @ V` path, a
   stacked component path, optional residual diagonals, and projected-core atom
   builders.
5. A randomized Nystrom-style sketch constructs the initial projected-core
   basis `U`; component core atoms are reduced while genotype blocks are
   streamed.
6. `fit_reml` fixes Hutchinson/SLQ probes and one projected-core SLQ reference
   factorization, caches `K_i @ Vrand`, and iterates PCG projection, score/AI
   evaluation, constrained Fisher steps, and line search. After each accepted
   nonterminal step, the model rebuilds only the PCG basis for the next round;
   the fit-wide SLQ reference remains unchanged. For one GRM plus an identity
   residual, it instead caches the Lanczos tridiagonal of `K` and evaluates
   every later `theta_g K + theta_e I` candidate by shift/scale of that small
   matrix.
7. Optional effect estimation solves for `P y`, derives SNP effects, and stores
   the phenotype transformation needed to return predictions to the original
   scale.

## Important Invariants

- Every operator must preserve shape and implement a symmetric PSD covariance.
  SMILE `W` symmetry/PSD is a trusted-input contract and is not checked at
  runtime.
- `diag_list[i]` must describe the same `K_i` as `K_mvs[i]`; its mean is used
  for initialization, preconditioning, and heritability. When external
  standardization statistics are supplied, the streamer measures the actual
  standardized column norms while building the cache. Ordinary GRMs expose
  this mean as a scalar rather than allocating an `n`-element constant vector.
- All sources in a multi-GRM fit must have identical aligned sample order.
- Prediction sources must reproduce training variant order, component layout,
  and standardization. Built-in BED/PGEN paths compare canonical
  `CHROM/POS/ID/REF/ALT` records from BIM/PVAR files; custom sources without a
  manifest remain responsible for equivalent validation.
- Fixed effects must be full column rank. High-level loaders add an intercept;
  low-level callers control the design matrix explicitly.
- REML standardizes its response unless `response_is_standardized=True`.
  Sparse COHERIT standardizes the phenotype once at entry and retains that
  scale through residual REML, LASSO, calibrated quadratics, and prediction.
- Sparse covariance parameters remain coefficients of the original kernels.
  Outer-loop h², final COHERIT h², and adaptive frozen-mean summaries use
  `sum_g theta_g * tr(K_g) / n` for background genetic variance. The trace
  atoms belong to the current fit's samples and partition; prediction keeps
  the training coefficients and standardization without rescaling the kernels.
- Strict REML requires converged PCG solves and symmetric operators. Approximate
  `smile_scoring` is a separate opt-in optimization policy.
- Sparse joint covariance blocks use the selected-mean information correction.
  They retain the
  Lasso residual's ordinary-covariate projection; profiling active SNPs as new
  ordinary covariates would implement a different estimator. Fixed-mean adaptive
  covariance-only fits keep the corrected K=1 sparse quadratic frozen.

## Performance-Critical Paths

- `K_i @ Vrand` is persistent state. Score traces use symmetry to avoid applying
  every `K_i` to all projected probes on every iteration.
- Single-GRM identity-residual SLQ uses Krylov shift/scale invariance. It scans
  the genotype operator once; strict line-search candidates still evaluate the
  full fixed-probe quadrature, but require only a small eigendecomposition.
- A single-GRM projected-core build reuses the Ritz values already produced by
  the randomized basis construction for `U.T @ K @ U`; it does not perform a
  redundant rank-wide genotype scan.
- Dense multi-GRM and admixed paths fuse compatible RHS/component work to reduce
  genotype scans and transfers.
- Sparse shard construction performs row correction sums with a bounded
  per-thread reduction and shifts shard column indices in place. It avoids
  temporary float arrays proportional to the total number of nonzeros.
- Multi-SMILE prediction decodes each shared genotype block once and multiplies
  all component effect columns together.
- Admixed exact diagonals scan the existing 2-bit cache directly and never
  materialize a wide standardized genotype block.
- A pinned ring overlaps host reads, H2D copies, and GPU matmul. A slot is not
  writable again until its asynchronous device transfer has completed.
- Dense streamers keep one call-shaped device copy of the standardization
  statistics. Streamed KV kernels donate the consumed accumulator buffer to
  their output, and PCG releases matrix-sized residual/work buffers as soon as
  their last operation has been dispatched.
- The dense AI bound subproblem is converted by Cholesky to NNLS and solved by
  SciPy's compiled active-set implementation. Its cost does not scale with one
  new JAX compilation per zero variance component.
- Preconditioner rebuilding happens only after a sufficiently large accepted
  parameter update, not during rejected line-search trials.
- The default runtime uses JAX's pooled allocator with preallocation disabled.
  The platform allocator remains available as an explicit diagnostic override.
- Adaptive scores use a common subspace and independent pilot/evaluation
  covariance-factor probes. A single LD prefix shares genotype products across
  boundaries. Only nuisance Fisher rows and evaluation variances are computed;
  the common quadratic matrices retain dependence between candidates directly.
  Large sketches and reference matrices are file-backed, and both projection
  and GPU integration use bounded batches. Temporary files are removed even
  on failure. Statistical and numerical assumptions are specified in
  [adaptive_score.md](adaptive_score.md).

## Extension Points

For a new covariance representation, implement the component `K @ V`, its
diagonal or mean-diagonal atom, and preferably batched `weighted_hv` and
`stacked_kv` functions. Supplying a projected-core atom reducer avoids
materializing `K_i @ U` stacks during preconditioner construction. Add a dense
small-`n` oracle test before enabling the operator in high-level pipelines.
