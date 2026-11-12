# Qwen3.5-9B instruct translation experiment

This study warms all seven existing defenders on unwatermarked English outputs
of `Qwen/Qwen3.5-9B`, then freezes that ensemble for all baseline and attack
comparisons. The primary detector is the existing equal-weight mean of member AI
probabilities. Individual detector results are diagnostics. No defender is
retrained on attack outputs in stages I or II.

The generator is the instruct checkpoint, with thinking disabled. The two
watermark arms are the released TextSeal 2 sampler and this repository's existing
DistSeal-inspired latent text teacher/student. TextSeal 2 changes sampling,
not model weights. The latent student is a merged model. This is not a claim of
an official DistSeal text reproduction.

## Frozen comparison

The existing CNN/DailyMail partitions provide pools of 1,000 training prompts and 500
validation prompts and a separate 5,000-article test cohort. Preparation rejects
repeated prompt IDs and repeated human continuations within or across groups.
The prepared run retains 984 unique training prompts and 488 unique validation
prompts after removing historical duplicates. Each task asks for a news continuation in a specified language, using the same
English article prefix as context. English instructions and context are retained;
this is direct foreign-language generation, not English output followed by a
round trip. Human continuations are fixed at 256 generator tokens.

Warmup reuses the existing paired 80/20 split and model-specific training workers.
Detection reports the ensemble at 0.5 and at a frozen threshold fitted to its
mean human validation score for nominal 1% FPR. The latter has finite-sample
uncertainty; test false-positive rates are reported separately. English test
outputs are never used for warmup, threshold selection or watermark selection.

For each clean, TextSeal 2 and latent arm, generate in Simplified Chinese,
French and Welsh. Translate each saved source into English independently using
`IndexTeam/Index-Translate-9B` and `google/translategemma-12b-it`. Use their
native translation prompts/templates and greedy translation. No watermark is
added by the translators. Generator sampling uses seed 42, batches of four,
temperature 1, top-p 0.95 and at most 256 new tokens. Translation allows 1,024
new tokens and retains truncation flags. Generation lengths are tokenizer
specific and do not represent equal information across languages.

Score source watermarks as well as translated watermarks, so failure to produce
a detectable source watermark is distinguishable from loss during translation.
TextSeal 2 uses the unchanged p <= 0.001 threshold. The latent threshold is
frozen from human and clean validation outputs. Empty and short English outputs
are retained and counted; valid detector evasion excludes outputs under 20 words.
Raw evasion is not a claim of useful, faithful or correctly translated English.
Language compliance and translation adequacy still need an independent audit.

MAUVE compares all three direct-English arms against the same 5,000 human
continuations. It uses response-only Mistral-7B final-token features, capped at
256 Mistral tokens, 25 clusters and seeds 42, 43 and 44. This custom-feature
metric is not comparable to default GPT-2 MAUVE. It measures distributional
similarity and alone cannot establish preserved quality. Feature blocks are
cached once; no generation is repeated to improve scores.

## Run and resume

Download the pinned snapshots listed in `scripts/translation_experiments.py`
using the authenticated Hugging Face environment before preparing. Both
translation templates must accept all requested language codes in GPU preflight.

```bash
PYTHONPATH=. uv run --no-sync python scripts/translation_experiments.py prepare \
  --warmup results/research_workflow/defender_comparison/base_pairs.json \
  --cohort results/research_workflow/grpo_quality/test5k-20261111T233619Z/cohort.json \
  --selection results/research_workflow/defender_comparison/notebook_selection.json \
  --quality-template results/research_workflow/grpo_quality/test5k-20261111T233619Z/unrestricted/manifest.json
sbatch --job-name=qwen9b-translation FROZEN_SOURCE/scripts/run_translation.slurm MANIFEST
```

The prepared run contains immutable source, partitions, model revisions and
hashes. Its sequential stages finish baseline training/scoring and English
MAUVE before translation. The launcher uses one H100, development QoS,
two-hour allocations and USR1 five minutes before timeout. Completed stages
are skipped. The existing training workers save optimizer, RNG and model state;
generation resumes at committed batch boundaries with a deterministic seed per
batch, and translation is greedy. This preserves the restart protocol, without
claiming cross-hardware bitwise determinism. The maximum is 16 automatic requeues
plus the initial allocation, an upper bound of 34 H100-hours. Reaching that bound
stops the job with its checkpoints retained. Worker failures are not blindly
requeued. No stage III defense training is in the stage list.

The CPU contract checks are:

```bash
PYTHONPATH=. uv run --no-sync python -m pytest ai_content_detector/tests/test_translation_experiments.py -q
```

## Initial full-paper reading, stage III remains deferred

Full PDFs were downloaded and read, including appendices. Local copies are
under the ignored watermark resources directory, in `translation_literature/`.

[He et al., ACL 2024](https://aclanthology.org/2024.acl-long.226/)
study generating in a pivot language and translating into the intended output
language. Their X-SIR defense combines multilingual context embeddings with
vocabulary clusters that assign related tokens a common bias. It improves
cross-lingual detection in the evaluated settings, but depends on dictionary
coverage and whole-word tokens. It is a separate watermark construction, not an
established repair for either TextSeal 2 or the local latent student. Our task
retains English prompts rather than translating the prompt as their CWRA does.

[Mohamed and Gubri, STEAM, arXiv v3](https://arxiv.org/abs/2510.18019)
propose detector-side back-translation search over candidate languages using
Bayesian optimization. They evaluate translator mismatch as well as non-English
source languages. This makes it a candidate for our unknown source language and
translator setting. Crucially, maximizing detection evidence across translations
requires calibration of the entire search statistic on independent human texts;
a single-translation cutoff is invalid. Their language/key-specific correction
for KGW is not a drop-in correction for TextSeal 2's Gamma test or our learned
extractor. Compatibility with these two watermark arms must be tested rather
than assumed. Search latency and weaker low-FPR performance with their small
local translator also matter.

[Zhou et al., LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.739/)
show that iterative adversarial training can recover detector robustness against
their word-substitution attacks. This supports considering translated-output
augmentation as a hypothesis, but does not establish transfer to unseen source
languages or translation models. Their evaluated back-translation baseline is
also different from generating foreign-language text directly.

After I and II, if an attack reduces detection on useful English outputs, a
possible detector study is augmentation across source-language/translator pairs
with entire languages, translators and their combinations held out in turn.
Include translated human controls to test whether a recovered detector merely
recognizes translation style. Evaluate the frozen ensemble, retain clean-English
performance and calibrate FPR separately. For watermarks, first test a bounded
STEAM-like detector-side search before changing the watermark itself. These are
proposed experiments, not implemented defenses or demonstrated robustness.

## Launch and validated checks

The 2026-11-12 run is `results/research_workflow/translation/20261112T000946297987Z/manifest.json`.
Its split audit passed with 984 training, 488 validation and 5,000 test examples.
All three pinned model snapshots downloaded successfully. The new translation
contract checks and existing instruct/latent watermark checks passed together:
22 tests, with two dependency deprecation warnings. This CPU run used the local
installed-package mirror and disabled optional GPU kernels in the test process.
The Slurm launcher passed `bash -n`; GPU preflight and experimental metrics are
pending. `results.md` updates after each completed stage.

Submitted Slurm job: `77713`, one H100 in development QoS. The scheduler reports
`QOSGrpGRES`, awaiting the QoS GPU quota. The current frozen manifest uses
5,000 examples for every translated condition as well as English MAUVE.
Its separate `compute_reservation.json` records the bounded allocation.

## First execution and memory correction

Job `77713` completed GPU preflight and all English baseline generations,
then failed during the first ensemble warmup member (Binoculars). Slurm reports
an OOM kill under a 40,000 MB host-memory allocation; this was not a timeout,
so automatic timeout requeue did not trigger. No detection-rate or MAUVE result
was produced. All 5,000 test generations remain saved, with no empty outputs or
outputs shorter than 20 whitespace-delimited words. Mean generation length was
169.34 tokens; 338/5,000 (6.76%) reached the 256-token cap. These are generation
diagnostics, not quality or detection measurements.

The launcher now requests 32 CPU cores to increase host memory. Jean Zay rejects
explicit `--mem` requests, so the resumed submission overrides CPU count on the
original frozen launcher. Source, partitions, decoding and model checkpoints stay
unchanged. Completed stages are skipped; warmup restarts from its available state.

Replacement job: `85527`, retaining one H100 and two-hour development allocations
with checkpoint requeue. Slurm accepted 160,000 MB of host RAM; the job is pending
on `QOSGrpGRES`. `resume_85527.json` records the allocation override and verified
frozen hashes. The launcher passed `bash -n` after the CPU-count change.

## Repeated warmup failure and preprocessing fix

Job `85527` also failed during Binoculars warmup after 8m40s, despite the
160,000 MB allocation. Increasing RAM did not resolve the failure. Neither run
reached dataset mapping progress or produced an optimizer checkpoint. Inspection
found that the dataset-map callback closed over `self`, allowing dataset
fingerprinting to serialize the entire two-model defender. The callback now
captures only its tokenizer and maximum length. A regression check guards against
serializing the defender during fingerprinting; restart tests validate the reused
training path. Full-model confirmation remains necessary.

`manifest-v2.json` and `source-v2/` retain this targeted fix separately from the
original frozen source. The revised manifest explicitly permits predecessor row
signatures and hashes the three completed baseline generation files. Original
source, partitions and generation rows remain unchanged. Completed generation
stages are reused, and the same seven-member ensemble is still required.

A CPU serialization reproducer confirmed that the original callback traverses
its captured defender, while a tokenizer-only callback serializes independently
(429 bytes for the lightweight reproducer). This supports the diagnosis, but
does not replace a successful real-model warmup.

The revised code passed 30 focused CPU tests across defender restart, defender
comparison and translation contracts, with two dependency deprecation warnings.
This includes the fingerprint regression and exact checkpoint-resume tests.
Job `87338` was submitted using `source-v2/scripts/run_translation.slurm` and
`manifest-v2.json`. Real-model confirmation and detection/MAUVE results are pending.

## Sanity audit after job 87338

The preprocessing fix passed real-model training: Binoculars and Fast-DetectGPT
both finished their three warmup epochs and saved `completed.pt`. Their internal
paired validation accuracies were 0.96954 and 0.99492, with AUROCs 0.99405 and
0.99979. These are per-member development metrics, not ensemble test results.
The seven-member ensemble is not yet ready, and no MAUVE result is available.

Job `87338` failed after 39m06s while saving a DivEye feature. The filesystem
had reached its 500,000-file limit, despite approximately 3.5 TB of free space.
All 1,548 cached feature files were consolidated into a SQLite database. Every
original payload was checked byte-for-byte against the database and a retained
`diveye_features_original.tar.gz` archive before removing the individual files.
No training outputs, model checkpoints or unrelated files were deleted.

DivEye now commits each new feature to the packed cache, with legacy `.pt` reads
supported. MAUVE feature blocks also use packed databases to avoid creating
another 2,500 small files. Texts, feature values, cohort sizes and model choices
are unchanged by these storage changes.

The audit found that `python -c` put the live checkout ahead of `PYTHONPATH`.
Before the cache patch, all non-test implementation files in the live checkout
matched `source-v2`; only two test files differed. Thus this audit found no
implementation drift in the completed training, but the import precedence was
incorrect. Workers now explicitly prepend the frozen source and reject already
imported live implementation modules. `manifest-v3.json` records the revised
source and accepts the declared predecessor row signatures. Existing generations
and the two trained ensemble members remain in place for resume.

The sanity revision passed 32 focused CPU tests (two dependency deprecation
warnings), including packed-cache reuse, native checkpoint resume and frozen
import precedence. Replacement job `92361` uses `manifest-v3.json`; no earlier
generation or trained member is intentionally repeated. `sanity_audit.json`
records preserved checkpoint hashes and cache verification. GPU confirmation
and full ensemble/test metrics remain pending.

## Cleanup and completed English baseline

Job `92361` completed warmup of all seven ensemble members and the fixed
5,000-sample English baseline. At the unchanged ensemble threshold of 0.5,
AI detection was 99.76% and human false positives were 0.26%. These are cohort
results; they do not establish robustness to translation. Saved per-condition
metrics are in `scores/base_en_metrics.json`.

The job then stopped before a latent watermark training checkpoint was saved:
some human reference responses were shorter than the 128-token channel. The
watermark-only training/selection partitions now exclude 13 training responses
and 3 validation responses, retaining 971 and 485 respectively. Detector warmup,
its scores, and every test row are unchanged. No padding or test filtering is
used. `latent_manifest-v4.json` records the separate input and a new protocol
hash; `manifest-v4.json` selects it and retains the predecessor source/inputs.

Under the user's cleanup authorization, 5,484 regenerable bytecode files with
confirmed source counterparts and 1,021 empty cache directories were removed
from the active environment. Installed source packages, model weights, datasets,
checkpoints and experiment outputs were retained. `uv cache prune --offline`
found no unused cache entries, and only the active managed Python installation
was identified, so no installed dependencies or interpreter were removed. This
freed approximately 6,500 file slots and 114 MiB. `cache_cleanup.json` retains
exact counts. Slurm workers already disable bytecode writes; validation also
uses `PYTHONDONTWRITEBYTECODE=1` to avoid repopulating these caches.

## Legacy environment removal and resubmission

The user confirmed that the portfolio-root environment was obsolete after moving
to `split-repos/`. Checks found no active Python or Slurm job using it, no checked
`.pth`/kernel reference from the active setup, and no split launcher reference.
The old root `.venv` was removed: 42,054 files/symlinks and 5,371 directories,
with 9,788,114,489 logical bytes of dependency files. The active split-project
environment and shared uv-managed Python were preserved. The filesystem then
had approximately 54,850 free file slots. Distribution metadata and exact cleanup
counts are retained in `legacy_environment_cleanup.json`.

The watermark input fix passed 25 focused CPU tests, with two dependency
deprecation warnings. Split-environment imports passed. Replacement job `96060`
uses `source-v4/scripts/run_translation.slurm` and `manifest-v4.json`, continuing
from completed ensemble training and baseline scoring. Watermark quality and
translation metrics remain pending.

After cleanup, Slurm accepted job `96060` and reported it RUNNING on H100 node
`jzxh001`. The old root environment is absent, the split-project environment
is present, and the shared managed Python remains installed.

## Two-H100 execution branches

The user requested that non-DistSeal work run on a second H100. The frozen
`manifest-v5.json` retains the same teacher/student training protocol, optimizer
state, cohort and detector checkpoints. The serial job is checkpointed before
replacement. `stages-distseal.json` and `stages-non_distseal.json` are disjoint:
the first completes the latent student and its evaluation/attacks; the second
runs TextSeal 2 and the clean-generator translation attacks independently.
Each branch requests one H100 with development-QoS checkpoint requeues.

MAUVE now accepts explicit arms: `quality:base:textseal2` runs before the latent
student is available, and `quality:base:latent` runs in its own branch. Both use
the same saved human cohort and feature protocol. Shared feature inserts retain
the committed database payload if both workers compute a block concurrently.
Partial MAUVE results merge under a file lock, summary publication is serialized,
and atomic JSON temporary names include the process ID. Thus parallel branches
do not overwrite each other's saved results. The branch split and concurrent
MAUVE-result merge passed the focused translation contract tests.

A measured window covered 100 student updates in 280.7 seconds: approximately
2.81 seconds/update. At logged step 2,850 of 8,000, linear extrapolation gives
about 4.0 additional H100-hours for distillation. A practical range is 3.5–5
hours of training compute. Checkpoint/export overhead, development-QoS queue
waits, generation, detector scoring, MAUVE and translation attacks are excluded.
The measurement is retained in `parallel_eta.json`; it is an estimate, not a
completion guarantee.

The safe handoff saved the student at update 2,893 before cancelling serial job
`96060`. DistSeal resumes in job `107004`; TextSeal 2 and clean-generator
translation work run in job `107002`. Both were observed RUNNING on distinct
H100 nodes (`jzxh004` and `jzxh279`). Ten focused translation tests and shell
syntax validation passed. Source-watermark scoring of the unwatermarked baseline
is explicitly marked not applicable in three receipts; its final-English
ensemble scoring is still scheduled. No watermark result is invented for that
clean arm.

At the user request, DistSeal job `107004` was checkpointed at student update
3,321 and placed in `PENDING (JobHeldUser)`, releasing its H100 allocation.
Job `107002` remains running. Resume of `107004` requires a later user request.

## User-requested 5,000-update DistSeal cap

After the hold at student step 3,321, the user requested a total of 5,000 updates
instead of 8,000. `manifest-v6-distseal5k.json` sets an explicit
`distillation_stop_step=5000`. The inner training protocol, saved optimizer/RNG,
and original 8,000-update cosine learning-rate schedule are retained. This is an
early stop of the same training trajectory, not a restart or a compressed
learning-rate schedule. There are 1,679 remaining updates, approximately 1.3
H100-hours at the last measured rate, excluding loading/export and queue waits.

The DistSeal-only manifest retains the predecessor quality-cache identity so
that base/human features remain reusable by the unchanged non-DistSeal job
`107002`. Latent features still validate their own generated-text hashes.
The previous held 8,000-update submission is replaced rather than editing the
shared manifest used by `107002`. Progress and export metadata record both the
5,000-update stopping point and the 8,000-update schedule length.

The cap revision passed 18 focused CPU tests, with two dependency deprecation
warnings. Job `111368` resumes from update 3,321 with the 5,000-update stop cap;
the superseded held job `107004` was cancelled after acceptance of its replacement.
Job `107002` and its frozen configuration were left untouched.

## First TextSeal 2 English results

Job `107002` completed the 5,000-sample TextSeal 2 English baseline and MAUVE.
Ensemble AI detection is 99.90%, with 0.26% human false positives at threshold
0.5. The private watermark detector flags 4,986/5,000 marked texts (99.72%)
and 3/5,000 human controls (0.06%) at the unchanged p <= 0.001 threshold.
MAUVE values for seeds 42/43/44 are 0.019717/0.020757/0.022332 for clean
outputs and 0.020297/0.019284/0.021207 for TextSeal 2. Similar, low values do
not alone establish preserved quality. Job `107002` is running the first
Chinese source-generation condition; translated-output detection is pending.

## English/Chinese 2,000-sample pilot on three H100 jobs

The user narrowed the study to direct English plus Chinese-to-English with all
three model versions and both translators. The fixed pilot is the first 2,000
IDs of the retained 5,000-example cohort: 6,000 Chinese source texts total and
12,000 translated outputs across six attack conditions. Existing output prefixes
are copied into `english-chinese-2k/`; the original run is preserved. Generation
continues from saved row counts rather than repeating completed text. Extra
saved English rows remain available for the later 3,000-example extension.
Receipts record their sample target so that a later 5,000-example run can resume
generation/translation beyond the pilot.

Generation now starts at batch 32 and translation at batch 16, with OOM fallback
to smaller batches before committing a result. New rows record batch size and
generation seed. Changing batching changes seeded sample trajectories for new
rows; existing texts are not replaced. Each case records timings and peak VRAM.
Allocations also record 10-second GPU utilization, VRAM and power telemetry.
Detector batches are configurable by member; score checkpoints are written every
128 outputs or 30 seconds, and immediately on a graceful stop or completion.
The original document-wise DivEye extraction remains unchanged for correctness.

Cached base/TextSeal 2/human probabilities are sliced to the same 2,000 IDs,
and the matching Mistral feature blocks are reused without GPU recomputation.
Their source hashes are recorded in `score_cache_import.json`. Model checkpoints,
score direction, calibration thresholds, token caps, sampling temperature/top-p,
watermark keys and the feature definition are unchanged. The cache import only
reuses saved probabilities for identical texts and trained checkpoints.

Jobs `116870` (base), `116871` (TextSeal 2), and `116872` (DistSeal) were submitted
on one H100 each in development QoS with exact checkpoint requeue. The revised
contracts and detector code passed 37 CPU tests, with two dependency deprecation
warnings; shell syntax and frozen hashes were checked. All three jobs were
accepted and initially pending on `QOSGrpGRES`. A provisional compute estimate
is 20–40 minutes after startup for initial comparisons and roughly 1.5–3 hours
for the full pilot; queue waits and large-batch throughput remain uncertain.

## French/Welsh 200-sample extension on one H100

Job `119154` submits a separate `french-welsh-200/manifest.json` and stage list
through the existing frozen `run_translation.slurm` launcher. It requests one
H100 in development QoS, with a two-hour allocation and the existing checkpoint
and automatic requeue mechanism. Slurm accepted it as pending on `QOSGrpGRES`.

The extension uses the same first 200 IDs of the original held-out cohort for
French and Welsh, all three existing model versions, and both Index-Translate
and TranslateGemma. This gives 1,200 source texts and 2,400 translated outputs
across 12 conditions. Generation/translation batches remain 32/16 with memory
fallback. The stage list contains no detector training, watermark training or
MAUVE recomputation. Existing English-warmed detector checkpoints, validation
thresholds, watermark models, and calibration records are reused. Human
watermark controls are the matching first 200 records. Source watermark scoring
precedes translation scoring for both marked models.

Preparation checked the manifest signature, all 1,671 frozen file hashes,
held-out split separation, 200 unique aligned IDs, 12 translation conditions,
and the required model/checkpoint files. All 13 focused translation-driver tests
passed in this checkout; the frozen Slurm launcher passed its shell syntax check.
The existing English/Chinese jobs and artifacts are unchanged. As in the Chinese study, requested source language
does not guarantee compliance; source-language validity and translation
adequacy must accompany interpretation of the raw detector/watermark rates.
