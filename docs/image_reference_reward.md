# Bounded SD1.5 reference reward

The original direct term mixed an unbounded KL loss with a PPO objective whose
image rewards were normalized into advantages. Its coefficient was not a reward
budget comparable to the evasion and CLIP weights.

The replacement uses the combined rollout reward

\[
R_i = 0.64 E_i + 0.26 C_i + 0.10 S_i,
\qquad S_i = \frac{1}{1 + \bar K_i / \tau}.
\]

Here `E` is detector evasion, `C` is the existing bounded CLIP alignment score,
and `S` is reference similarity. All three are in `[0, 1]` before weighting and
joint advantage normalization. Their maximum contributions are respectively
`0.64`, `0.26`, and `0.10`; there is no additional direct KL penalty.

For image `i`, the raw statistic is

\[
\bar K_i = \frac{1}{T d}\sum_{t\in\mathcal T}
\frac{\|\mu_\theta(s_{i,t})-\mu_0(s_{i,t})\|_2^2}{2\sigma_t^2},
\]

where $d$ is the latent dimension and $T=|\mathcal T|$ counts stochastic transitions,
and the two Gaussian transition means use the same state, prompt, guidance and
variance. The initial SD1.5 reference is evaluated with LoRA disabled. The
terminal deterministic transition is excluded. The statistic is per latent
coordinate and transition, not KL between final image distributions.

The dedicated experiment selects `tau` from the median positive per-image KL
measured after one unanchored probe epoch on eight training prompts. The probe
uses the same learning rate, sampling schedule and update rule as the main
attacker. Its weights are discarded; the main 50-epoch attacker starts again
from initial SD1.5. No validation or evaluation prompts choose the scale.
The selected median maps to reference similarity `0.5`, while zero drift maps
to `1`. The scale is then frozen and stored with the effective configuration.
Invalid measurements or no measurable probe drift stop the job rather than
silently selecting a scale.

The KL-derived score is detached rollout reward shaping. It is not an exact
KL-regularized objective or a hard KL constraint. Bounding the contribution
prevents unbounded loss magnitude; it does not guarantee equal component
variances or gradient influence. The report retains raw KL, component means,
ranges, standard deviations and weighted variations in `reward_scales.csv`.

The generic DDPO configuration exposes `reference_kl_scale` explicitly. The
training-only calibration described here is performed by the dedicated image
experiment driver. The legacy direct `kl_coeff` defaults to zero and cannot be
combined with a positive reference-reward weight.

The replacement manifest adopts the existing preparation, benchmark and warm-up
jobs. Only pending attacker and defense jobs are replaced. Their new held jobs
are validated against the existing warm-up dependency before the old downstream
jobs are canceled and the replacements released. Defense also waits for the
existing benchmark, so the final replacement report includes both branches.
