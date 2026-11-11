# Watermarking and removal

This package separates watermark creation, removal attempts and owner-side
evaluation. It covers the official TextSeal 2 sampler, released DistSeal image
models and an experimental latent-text adaptation. These are different
watermark locations and threat models; see the [protocol](../../tex/records/docs/watermarking.tex)
for their exact relationship and current experiment status.

## Read the implementation

| File | Responsibility |
|---|---|
| [protocol.py](protocol.py) | Stage definitions, partitions, thresholds and metric contracts |
| [text.py](text.py) | TextSeal integration and text evaluation/distillation helpers |
| [latent_text.py](latent_text.py) | Experimental embedder/extractor and discrete text channel |
| [image.py](image.py) | Image generation and watermark evaluation |
| [backbones.py](backbones.py) | Native released model backbones and checkpoint state |
| [removal.py](removal.py) | Attacker tasks, artifact integrity and owner evaluation gating |
| [preflight.py](preflight.py) | Resource and environment checks before experiments |
| [text_attacks.py](text_attacks.py) | BIRA, resampling, smoothing, source blocking and ratio clients |
| [copula_exposure.py](copula_exposure.py) | Conditional TextSeal 2 probability estimation |
| [exposure_support.py](exposure_support.py) | Retained-mass optimization and native evidence bounds |
| [preservation.py](preservation.py) | Aligned quality gates, blind selection and joint removal accounting |
| [image_quality.py](image_quality.py) | Matched image metrics and source integrity |
| [semantic_similarity.py](semantic_similarity.py) | Paired text embeddings and uncertainty |

Start with the protocol and removal contracts before following an individual
training worker. [watermark_experiments.py](../../scripts/watermark_experiments.py)
and [watermark_removal.py](../../scripts/watermark_removal.py) prepare manifests
and frozen source snapshots; their commands are in the protocol above.
The separate [TextSeal API protocol](../../tex/records/docs/textseal_api_attacks.tex) describes
blind rewriting and simulated probability access on the original sampler.

## What the comparison means

Ordinary sampling can bypass a sampler-only watermark without changing model
weights. A watermark distilled into weights requires a different removal test.
Tasks with original clean weights can interpolate with that reference; tasks
without those weights use clean-data updates on the watermarked model. Owner
keys, watermark detector outputs and clean reference weights are excluded from
those no-reference worker contracts.

Those contracts separate process arguments and hash fixed artifacts; they are
not an operating-system sandbox against a malicious process on the same shared
account. The owner evaluates only after attack artifacts have been frozen.
Validation and final evaluation remain separate from clean-data fitting.

## Current text experiments

Use the [instruction-model protocol](../../tex/records/docs/instruct_watermarking.tex) and
`scripts/instruct_watermark_experiments.py`. Both TextSeal 2 and the stronger
DistSeal-inspired student use Qwen/Qwen3.5-4B instruct. BIRA uses only
Llama-3.1-8B-Instruct. Earlier Base-model watermark results were deleted.
`instruct_text.py` builds native chat prompts, disjoint question groups and
released TextSeal 2 configuration/scoring. `text.py` masks prompt predictions
out of student training; `latent_text.py` aligns the teacher channel with
response prediction positions. Quality diagnostics do not block removal.

```bash
PYTHONPATH=results/research_workflow/watermarking/resources/textseal2:. uv run --no-sync python -m pytest ai_content_detector/tests/test_instruct_watermarking.py ai_content_detector/tests/test_latent_watermarking.py ai_content_detector/tests/test_bira_replication.py -q
```

[Jina v5 similarity](../../tex/records/docs/jina_similarity.tex) compares all text attack
outputs by prompt ID using `semantic_similarity.py` and
`scripts/jina_watermark_similarity.py`. It reports continuous cosine scores
and uncertainty alongside the existing NLI and quality diagnostics.

The [matched image-quality extension](../../tex/records/docs/matched_image_quality.tex) uses
`image_quality.py` and `scripts/compare_image_attack_quality.py` to evaluate
all saved weight and NFPA outputs with the same paired and class-level metrics.

For repository publication, use the [self-contained results report](../../docs/watermarking/README.md).
Validate the tracked tables, separate TeX records and eight-pair galleries with
`PYTHONPATH=. uv run --no-sync python scripts/publish_watermark_results.py --check`.
Validation uses no local model/data artifacts and is also the default action.
Historical Markdown export remains available with `--legacy-markdown --output
LEGACY_EXPORT_DIRECTORY`; it cannot overwrite the canonical documentation.

The [pedagogical report](../../tex/report.pdf) uses [modular LaTeX sources](../../tex/report.tex). Research-document
snapshots and their separate TeX records are linked in
`tex/provenance/migration.json`; operational Markdown and frozen experiment
snapshots are retained.
`preservation.py` joins frozen owner decisions with explicit quality constraints;
raw evasion and joint content-preserving removal have separate denominators.
The post-hoc text and image audits are `tex/quality_audit.py` and
`tex/image_preservation_audit.py`. Run them from the repository root with
`PYTHONPATH=. uv run --no-sync python <script>`.

`survival.py` and `scripts/watermark_survival_pilot.py` implement a development
hypothesis for redistributing an existing watermark task vector. This is not an
established defense or novelty claim. The pilot freezes inputs and code, uses
matched generation seeds and a preservation screen, and saves completed work
before a dev-QoS requeue. A passing pilot alone does not establish attack robustness.
The preservation and filter checks passed in this checkout:
`PYTHONPATH=. uv run --no-sync python -m pytest -q ai_content_detector/tests/test_preservation.py ai_content_detector/tests/test_watermark_survival.py`.

The survival-filter pilot failed all initial quality and detection gates.
Its negative result and paired images are retained in the pedagogical report.
The subsequent development study compares fixed and adapted clean-reference teachers with
`scripts/watermark_transport_pilot.py`, `scripts/watermark_transport_train.py`,
and `scripts/watermark_transport_screen.py`. The teacher check passed the frozen
screen: both teachers had 32/32 detections and 31/32 preservation passes.
Meta-distillation and independent removal evaluation are separate stages. The
worker implements a first-order approximation through detached Adam updates,
not exact differentiation through the optimizer.

`copula_exposure.py` estimates the key-conditional TextSeal2 race distribution by
integrating each lane's own Gaussian noise. Its normalized finite-sample estimate
has bias; it is an experimental replacement for pre-watermark probability
exposure, not a claim about an unchanged API. The analytic two-lane probability,
zero-probability support and independent/deterministic limits passed two checks:
`PYTHONPATH=. uv run --no-sync python -m pytest -q ai_content_detector/tests/test_copula_exposure.py`.
`scripts/textseal_exposure_pilot.py prepare` freezes fresh questions and a blind
rewriting protocol. The generation stage retains every attempt; quality-based
selection and frozen owner scoring must finish before reporting removal success.

New jobs use the matching `run_*` dev Slurm scripts with rolling checkpoints and
bounded requeues. The total budget and reservations are in
`tex/provenance/compute_budget.json`. An exact mirror of the installed Transformers
and Diffusers package files is unpacked on the node to reduce Lustre startup
scans, without changing the installed environment.

## Development results and reproduction

Run the commands below from the repository root. Full numeric results and their
scope are in the [pedagogical report](../../tex/report.pdf).

The image prediction-anchor variants pass their initial 32-sample quality
screen, but the stronger quality-fallback attacker yields 20/64 joint removals
for the original model and 22/64 for both trained variants: no defense gain.
The matched 500-inner-step revision also fails to improve robustness: joint
removals are 20/64 original, 21/64 fixed-reference and 20/64 adapted-reference.
This meta-distillation direction is stopped under its frozen criterion.
After qualitative review, the current exploratory quality rule uses LPIPS at
most 0.2, image CLIP at least 0.9 and class-alignment retention at least 0.9.
SSIM remains a reported metric but no longer gates preservation: all nine
SSIM-only exclusions retain the main scene under the user's tolerance for minor
appearance changes. With this rule, the 500-inner-step comparison gives 29/64
quality-preserving removals for the original model and 28/64 for each variant.
Paired uncertainty intervals include no difference. This is post-hoc reanalysis,
not fresh confirmation or independent human-quality validation. See
`tex/provenance/revised_quality_rule_no_ssim.json` and
`tex/provenance/ssim_ablation_audit.json` for the frozen rule and full audit.
Conditional probability
exposure reduces valid development removals from 25/32 to 5/32 for direct
resampling and from 22/32 to 12/32 for smoothing. However, adaptive ratio
reweighting removes all selected watermarks under both interfaces; fewer joint
removals under conditional exposure reflect quality failures, not robustness.
The completed 128-question confirmation reduces joint direct-resampling removals
from 114/128 to 19/128, but conditional ratio reweighting still removes 126/128
watermarks, including 95 of its 96 quality-preserving outputs. A separate
support-pruning development test reduces conditional-ratio joint removals from
21/32 to 3/32 or 1/32 while increasing quality passes from 21/32 to 25/32 or
24/32. Independent rewriting still evades on all 22 quality-preserving outputs,
and smoothing does not improve. This restricted-client gain is not independent
confirmation or evidence of a generally robust watermark.
The report also covers the separate 4,096-prefix human-null audit, which
flagged 5 prefixes at the unchanged nominal `p ≤ 0.001` threshold.
The audit is conditional on the evaluated key and HC3 cohort.

For the quality-aware image experiment, prepare validation from the completed
attack manifest, then run each declared case using its frozen dev job script:

```bash
PYTHONPATH=. uv run python scripts/watermark_preserving_evaluate.py prepare ATTACK_MANIFEST
sbatch FROZEN_SOURCE/scripts/run_watermark_preserving_evaluate.slurm EVAL_MANIFEST validate original
sbatch FROZEN_SOURCE/scripts/run_watermark_preserving_evaluate.slurm EVAL_MANIFEST validate fixed
sbatch FROZEN_SOURCE/scripts/run_watermark_preserving_evaluate.slurm EVAL_MANIFEST validate adapted
```

Owner evaluation uses the same script with `EVAL_MANIFEST owner` only after all
three quality-only selections are frozen.

The discarded LoRA retraining pilot produced zero removals on all three models.
The report and [compact provenance record](../../tex/provenance/discarded_lora_run.json)
retain the negative result.
The development utility retains `--retrain-lora` support for native merged
exports and resumable adapter training.

A stronger text rewrite pilot adds
`--block-source-ngrams` to `scripts/textseal_exposure_pilot.py prepare`; it records
fallbacks if source-pair blocking leaves no probability mass. Reused sources are
development data, requiring fresh confirmation before a final defense claim.

`--ratio-attack` adds the exposed-support surrogate-ratio client.
`--audit` runs the finite-budget probability and native-clipping audit;
`--audit-attack-prefixes` uses fixed positions from saved stronger rewrites.
Auditing saved attack prefixes also requires `--reuse-parent`.
Fresh confirmation uses `--confirmation-shard 0` and `--confirmation-shard 1`,
each with 64 disjoint questions, six attack arms and a maximum of eight resumable
30-minute dev allocations. Reserve their maximum compute in the shared ledger
before submission.

`--support-guard --block-source-ngrams --reuse-parent BLOCKED_MANIFEST` prepares
the separate support-pruning development experiment. It keeps 90% or 80% of the
ordinary advertised top-ten conditional mass while maximizing the minimum native
token evidence. Retained probability estimates are unchanged; fewer alternatives
may be returned. The guarantee covers choices within the returned support.
Reference smoothing and independent rewriting are explicit controls that can
leave it. The objective and its limits are explained in
[the support-pruning section](../../tex/sections/defense_support.tex).

Generate development tables and paired qualitative evidence from completed owner
results with `PYTHONPATH=. uv run --no-sync python tex/development_summary.py text QUALITY_MANIFEST`
or the same command with `image EVALUATION_MANIFEST`.
Combine the two disjoint confirmation quality manifests with
`PYTHONPATH=. uv run --no-sync python tex/confirmation_summary.py SHARD0_QUALITY_MANIFEST SHARD1_QUALITY_MANIFEST`.
This checks source and calibration separation and produces paired uncertainty,
exact binomial intervals, a three-panel figure and the shard-level full texts.
