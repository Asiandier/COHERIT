# Joint split-GRM score test

The score stage fits no alternative model for individual cuts. It evaluates
efficient covariance scores under the current null and calibrates their maximum
absolute standardized value using one common Gaussian quadratic reference.
There is no phenotype resampling or repeated fitting inside the reference
integration.

## One split decision

1. Use the current partition, covariance estimate and frozen sparse mean.
2. Evaluate every remaining testable point in the predetermined LD-rank grid.
3. Remove the current GRM and residual-variance nuisance directions.
4. Compute `T = max_t |U_eff(t)| / sqrt(I_eff(t))`.
5. Compare T with the joint reference maximum. Add its maximizing boundary
   only when the joint p-value is at most `--split-alpha`.

All current parents' candidates belong to one family. Ties in absolute score
are resolved by smaller boundary position. Failure of numerical precision is an
error, not evidence of homogeneity. A zero-variance parent has no interior
two-sided split direction and is excluded. A direction with zero efficient
information is also excluded and recorded.

## Null model and score

For `r = y - frozen_sparse_mean`, the working Gaussian null is

$$
r=C\gamma+\epsilon,\qquad
V=\operatorname{Cov}(\epsilon)=\sum_g\theta_gK_g+\theta_eI,\quad\theta_e>0.
$$

For a proposed left/right split of parent J, use the actual GRM normalizers:

$$
K_J=\frac{m_L}{m_J}K_L+\frac{m_R}{m_J}K_R,\qquad m_J=m_L+m_R.
$$

Homogeneity means `theta_L/m_L = theta_R/m_R`. Parameterize its departure by
`theta_L = theta_J*m_L/m_J + delta` and
`theta_R = theta_J*m_R/m_J - delta`. Thus `dV/d(delta) = D_t = K_L-K_R`.
The normalizers are the actual `m_eff` values; neither unit mean GRM diagonal
nor equality of the raw left/right coefficients is assumed.

Define

$$
P=V^{-1}-V^{-1}C(C^\top V^{-1}C)^+C^\top V^{-1}.
$$

Then `PC=0` and `PVP=P`; generally `P² != P`. Differentiating the REML
log likelihood along any symmetric covariance direction D gives the unscaled
score and Fisher inner product

$$
U(D)=\tfrac12\{r^\top PDP r-\operatorname{tr}(PD)\},\qquad
F(D,E)=\tfrac12\operatorname{tr}(PDPE).
$$

Under a fixed correct null, `E U(D)=0` and `Cov(U(D),U(E))=F(D,E)`.

Let N denote the existing GRMs and identity residual direction. Project in the
Fisher inner product:

$$
c_t=F_{NN}^{+}F_{ND_t},\quad
\widetilde D_t=D_t-\sum_a c_{ta}N_a,\quad
\widetilde U_t=U(D_t)-c_t^\top U_N,\quad
v_t=F(\widetilde D_t,\widetilde D_t).
$$

Observed nuisance scores remain in the adjustment; a numerical REML stopping
condition does not imply that they are exactly zero.

For efficient computation the code uses global prefixes
`D_t = Z_low Z_low'/t - Z_high Z_high'/(M-t)`. Within each current parent,
the difference from a positive multiple of the parent-local direction has
constant marker weight and therefore belongs to the current nuisance span.
Exact Fisher projection removes that difference, leaving the same standardized
score. Existing GRM normalization and trace-weighted heritability are preserved.

## Why the reference is quadratic

Since `Pr ~ N(0,P)` under the fixed correct null, all standardized scores have
the joint representation

$$
B_t=\frac{P^{1/2}\widetilde D_tP^{1/2}}{\sqrt{v_t}},\qquad
Z_t=\tfrac12\{g^\top B_tg-\operatorname{tr}(B_t)\},\quad g\sim N(0,I).
$$

All candidates share g. Each marginal is a signed sum of centered chi-squares,
which need not be Gaussian when LD concentrates the spectrum. A Gaussian vector
with matching covariance would not preserve these tails.

The ideal p-value is

$$
p=\Pr_g\!\left\{\max_t
|\tfrac12(g^\top B_tg-\operatorname{tr}B_t)|\ge T_{\rm obs}\right\}.
$$

The maximum already accounts for searching the given candidate family. No
additional multiplicity adjustment is applied to the same family.

## Common core and covariance-factor trace probes

Build `H=PT` with `T'PT=I` from a randomized root-GRM subspace independent of the
response under the fixed null. Then `P_perp=P-HH'` is positive semidefinite.

Generate independent covariance-factor Rademacher probes

$$
f_i=\sum_g\sqrt{\theta_g/m_g}\,Z_gs_{gi}+\sqrt{\theta_e}\,s_{ei},
\qquad E(f_if_i^\top)=V,
$$

and compute `b_i=P f_i-H(H'f_i)`. Their covariance is `P_perp`.
Let `B=[b_1,...,b_R]` and, for each covariance direction D, form

$$
A=H^\top DH,\quad E=H^\top DB,\quad C=B^\top DB.
$$

If `C°` denotes C with zero diagonal, the common small quadratic matrix is

$$
M_D=
\begin{pmatrix}
A & E/\sqrt R\\
E^\top/\sqrt R & C^\circ/\sqrt{R(R-1)}
\end{pmatrix}.
$$

Its reference is `(xi' M_D xi - tr(M_D))/2`. The cross and bulk blocks share the
same probe coordinates of xi, so they preserve their joint quadratic structure.

The trace and Fisher estimates are

$$
\widehat{\operatorname{tr}(PD)}
=\operatorname{tr}(A)+\frac1R\sum_i C_{ii},
$$

$$
\widehat F(D,E)
=\tfrac12\langle A_D,A_E\rangle_F+
\frac{\langle E_D,E_E\rangle_F}{R}+
\frac{\langle C_D^\circ,C_E^\circ\rangle_F}{2R(R-1)}
=\tfrac12\operatorname{tr}(M_DM_E).
$$

For fixed directions, Fisher is unbiased and is a positive semidefinite Gram
by construction. Only its nuisance rows and evaluation diagonal are needed.

Independent pilot probes fit `c_hat`. Evaluation probes then measure the actual
direction `D_bar=D-N c_hat`. They do not fit another Schur complement.
Both the observed score and the reference use the evaluation information.
The common matrix is divided by `sqrt(v_hat)` before integration.

For fixed dimension and a fixed candidate family, the probe covariance
`BB'/R` converges to `P_perp`. The centered diagonal correction removed from
the reference has variance tending to zero under finite fourth moments.
With identifiable nuisance information and nonzero candidate information, this
gives convergence of the joint reference as both probe groups grow. It is not
a finite-probe calibration certificate.

## Numerical precision and memory

Pilot starts at `min(max_probes, max(initial_probes, 4*nuisance_count))`.
Evaluation begins at initial_probes and doubles until, for every testable cut,

- score-centering trace SE / sqrt(information) <= trace tolerance;
- information SE / (2*information) <= trace tolerance.

Information SE uses the combined delete-one jackknife of cross and off-diagonal
bulk terms, including their dependence. Core terms are fixed during that
jackknife. SE thresholds are numerical diagnostics, not simultaneous confidence
intervals. Probe growth uses these diagnostics, not the observed p-value.

The score stage uses PCG tolerance no larger than `min(pcg_tol, 1e-5,
score_trace_tol/10)` and checks true residuals. The GRM sketch is orthogonalized
before its P Gram is whitened.

Factor generation, P solves and marker projections use bounded batches.
LD contractions retain one running prefix. Arrays larger than 8 MiB use
temporary memory-mapped files beside the score output; projected directions and
reference integration also use bounded batches. The scratch directory is removed
on both success and failure. Disk storage still scales with the probe caches
and the number of candidates times the squared reference dimension; this
approach bounds working batches, not the total storage requirement.

## Integration, parameters and outputs

The default shared core rank is 64. There are initially 512 probes in each
independent group, a cap of 4096 per group, tolerance 0.05, and a fixed 16,383
reference samples. Parameters are:

- `--score-core-rank`
- `--score-trace-probes` and `--score-trace-max-probes`
- `--score-trace-tol`
- `--score-reference-samples`
- `--score-trace-seed`

One seed deterministically produces independent core, pilot, evaluation and
integration streams. Each candidate batch replays the same Gaussian integration
coordinates. Float32 GPU products use JAX's highest multiplication precision;
host trace/Fisher contractions use float64.

For reference maxima `T_1,...,T_N`, report

$$
\widehat p=\frac{1+\#\{i:T_i\ge T_{\rm obs}\}}{N+1}.
$$

N is fixed before examining the tail count and must resolve split-alpha.
The rank p-value is finite-Monte-Carlo valid when the observed statistic and
reference statistics are exchangeable draws from the exact null. Fitted V and
finite traces leave the actual implementation a plug-in test; the +1 formula
does not eliminate those approximations.

Score JSON schema 3 has method `joint_quadratic_reml_ld_cusum` and criterion
`global_p_value_le_split_alpha`. Candidate rows report raw and efficient scores,
efficient information, signed `standardized_score` and squared `score_statistic`.
Diagnostics include `global_p_value`, `maximum_absolute_score`,
`reference_samples`, `reference_seed`, `reference_tail_count`,
`reference_critical_value`, core/probe sizes, SEs and untestable directions.
The existing path-history field `score_p` now carries the joint p-value.
An old score schema is rejected; the run signature also records the source digest
and new parameters.

## Guarantee and interpretation

For a fixed correct Gaussian mean/covariance model, a fixed candidate family,
and exact matrices and calibration, the probability of falsely deciding to
split this family is at most alpha. Under a feasible covariance change
`V_delta=V+delta D_t`, with P held at the true null,

$$
E_\delta[\widetilde U_t]=\delta v_t,\qquad
E_\delta[Z_t]=\delta\sqrt{v_t}.
$$

This explains the signal scale and the role of efficient information. It does
not imply equal power in all LD structures or universal optimality.

The production covariance is fitted, and the frozen sparse mean is estimated.
Correct mean specification and suitable estimation/numerical error conditions
are needed for the working-null approximation. A selected boundary is a location
estimate, not a confidence statement about its exact position. A single-family
test is not a theorem about the final number of segments after repeated use of
the same observations.

## Validation

`tests/test_score_process.py` checks dense REML/Fisher identities, nuisance
projection, the joint reference covariance and jackknife, degenerate directions,
known chi-square tails, duplicate candidates, and batching invariance.
`tests/test_adaptive_ld.py` checks genotype normalization, source/cache order,
covariance-factor probes, temporary cleanup, and real BED/PCG scores against
a full dense reference. The adaptive end-to-end test exercises split decisions,
refits, final output, prediction and resume.

The reproducible offline experiment runs the production operator construction
and calibrator against a complete quadratic reference:

```bash
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tests/validate_score_process.py --out /tmp/score-validation.json
```

Four Gaussian genotype models use n=128 or 512, LD correlation 0 or 0.9, and
127 or 31 candidate cuts. Each has 30,000 null and 30,000 alternative datasets
and two trace seeds. The alternative assigns left/right genetic coefficients
0.49/0.01 with residual coefficient 0.5. Reports include Wilson intervals,
probe counts, actual trace/information errors, and exact-reference comparisons.
The two trace seeds reuse the same data within each model and must not be
pooled as independent datasets. These experiments test a known mean/covariance;
they do not certify the complete adaptive path.

The integrated implementation's 2026-09-07 run produced the following rates
(nominal alpha 5%; two trace seeds shown separately):

| n, m, LD correlation | Cuts | Full-reference null rejection | Trace-reference null rejection | Full-reference power | Trace-reference power |
|---|---:|---:|---:|---:|---:|
| 128, 256, 0 | 127 | 5.160% | 5.080%, 5.330% | 25.660% | 25.023%, 26.513% |
| 128, 256, 0.9 | 127 | 4.907% | 4.753%, 4.820% | 17.643% | 17.233%, 17.383% |
| 512, 512, 0 | 31 | 5.090% | 4.867%, 5.137% | 99.843% | 99.857%, 99.830% |
| 512, 512, 0.9 | 31 | 4.833% | 4.863%, 4.987% | 29.043% | 29.083%, 30.580% |

This run used 512 probes per group in every case. The 4.753%–5.330% range
shows the finite-probe/integration variation; it is not evidence of an exact
finite-sample 5% bound. All figures can be regenerated with the command above.
