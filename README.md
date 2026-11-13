# AI Content Robustness

A research workbench for AI-content detection, adaptive text and image generation,
and watermark persistence under removal attempts. It studies how detectors
generalize, whether retraining recovers detection after an adaptive attack, and
how attacker access and content preservation change watermark-removal results.
A Streamlit application exposes the detectors for interactive scoring.

The [pedagogical report](tex/report.pdf) connects the methods, mathematical
definitions and experimental results. The [watermarking results](docs/watermarking/README.md)
include complete metric tables and fixed-sample comparisons. The
[detector catalogue](DETECTORS.md) explains the implemented methods and their
score conventions.

## Selected results

The supervised benchmarks and adaptive experiments use separate protocols.
The text benchmark reports Qwen macro F1 of 0.9236 on MAGE and 0.8310 on Pile,
while Gemma reaches 0.9499 on peer reviews. In the image benchmark, EfficientNet
macro F1 falls from 0.9346 on ArtiFact to 0.3333 on CIFAKE. A ConvNeXt augmentation
intervention fails its exploratory acceptance criterion. Dataset shift therefore
matters even for strong in-distribution classifiers. The report retains the full
model comparisons and evaluation boundaries.

### Adaptive generation and defender recovery

Qwen3.5-9B-Base LoRA attackers target three defenders. Each branch evaluates 500
generated texts and 500 human passages. Evasion is the fraction of generated
texts with AI score below 0.5.

| Target | Attacker updates | Evasion before defender update | After one defender update |
|---|---:|---:|---:|
| Gemma | 1,213 | 77.2% | 0.2% |
| ModernBERT-large | 1,487 | 96.8% | 1.0% |
| Seven-detector ensemble | 502 | 94.8% | 0.4% |

Before and after use identical saved texts. Defender fitting uses fresh training
outputs. Training budgets differ across branches, and another adaptive attacker
round remains untested. These results establish recovery against the saved
attackers. The [experiment record](tex/records/docs/research_results.tex) retains
the detailed measurements.

The saved ensemble outputs can also be compared with the base generator using
`scripts/compare_grpo_quality.py MANIFEST`, launched with
`sbatch scripts/run_grpo_quality.slurm MANIFEST` on one H100 development GPU
(30-minute limit). The manifest identifies the 500 aligned before/after samples,
frozen local Mistral-7B and E5 models, and runtime archive. Results include
prompt-conditioned, token-weighted generation perplexity, mean per-document
empirical token entropy in bits, E5 similarity to full human references for both
sets, and MAUVE against the saved baseline human continuations. MAUVE uses custom
Mistral final-token features, 25 buckets and seed 42; it is not directly comparable
to default GPT-2 MAUVE scores. These are automatic quality and diversity proxies.
The optional research dependencies include `mauve-text==0.4.0`. Run artifacts
and per-sample measurements live under `results/research_workflow/grpo_quality/`.

On 2026-11-12, H100 development job 76441 completed this comparison on all
500 paired samples. Before/after GRPO, generation perplexity was 3.5509/7.9577,
mean token entropy was 6.0434/6.8394 bits, E5 similarity was 0.8576/0.8701,
and custom-feature MAUVE was 0.0992/0.7492. Human continuations had perplexity
5.6374 and token entropy 6.9258 bits. Perplexity therefore worsens while these
similarity and diversity measures improve; these results do not establish an
overall quality gain. The baseline E5 score is now computed, and the recomputed
after-GRPO score agrees with the retained evaluation. Full protocol and metrics:
`results/research_workflow/grpo_quality/20261111T230402Z/results.json`.

A separate constrained comparison uses `scripts/generate_constrained_grpo.py`
and `scripts/run_constrained_grpo.slurm`. It regenerates all 500 prompts with
both the base model and ensemble checkpoint 502, forbidding `<think>`, `</think>`
and `OPTIONS:` token sequences, including leading-space and newline encodings.
Both models use seed 42, temperature 0.8, top-p 0.95, 256 new tokens and batches
of eight. Human controls remain the original saved continuations. The original
quality results remain unchanged; `comparison.md` in the constrained run places
the original and new metric tables together. This changes decoding only and
does not retrain the attacker or establish its constrained detector evasion rate.

The 500-sample constrained job 76910 was cancelled before completion. Its
replacement uses `scripts/prepare_grpo_quality_5k.py` to freeze 5,000 unique
CNN/DailyMail test articles, explicitly excluding all recorded attacker training
and evaluation prompts. Each article contributes its first 50 words as prompt
and the remaining human text as semantic reference. MAUVE uses the same 5,000
human continuations capped at 256 Qwen tokens for both models and both decoding
regimes. Separate unrestricted and constrained jobs use batches of 32, 256 new
tokens, temperature 0.8, top-p 0.95 and seed 42. Each requests one H100 in the
development QoS for up to two hours. MAUVE retains the custom Mistral features
and 25 clusters, with clustering seeds 42, 43 and 44 recorded separately. More
samples and seed checks do not establish universal metric reliability. The
historical 500-sample table is preserved separately; it uses a different cohort
and human-length rule. The new 5,000-sample tables compare the same fixed test
cohort. Manifests, overlap audit and tables are retained under
`results/research_workflow/grpo_quality/test5k-20261111T233619Z/`.

The first 5,000-sample constrained run (77129) failed its literal-string audit:
155 baseline outputs contained `OPTIONS:` through alternate tokenization such
as `OPTIONS://`; no thinking tags survived. These outputs are excluded from the
quality comparison. The corrected `scripts/forbidden_text.py` processor blocks
literal-string completions across alternate token boundaries, in addition to
the original token-sequence bans. The corrected run is retained separately in
`test5k-20261111T233619Z/constrained-v2/`. Its generation cache is independent of
the failed run. The focused regression tests cover alternate suffix tokens,
split thinking tags, whole-string tokens, and unrestricted scores.

Both 5,000-sample jobs completed successfully: unrestricted job 77125 in
1h11m20s and corrected constrained job 77744 in 1h18m50s. Post-run validation
confirmed 5,000 aligned outputs per model/regime, the same human controls,
consistent perplexity aggregation, and zero occurrences of the three forbidden
strings in either constrained set. At clustering seed 42, unrestricted
before/after GRPO perplexity is 4.2153/7.1682, entropy 6.3101/6.8600 bits,
E5 similarity 0.8598/0.8764, and MAUVE 0.3340/0.8772. With constraints these
are 4.5151/7.1609, 6.3237/6.8594 bits, 0.8565/0.8764, and 0.2865/0.8803.
The human controls have perplexity 5.8132 and entropy 6.9632 bits. Combined
historical and new tables are in
`results/research_workflow/grpo_quality/test5k-20261111T233619Z/comparison.md`,
with counts, literal-string audits and MAUVE seed variation in `validation.json`.
The forbidden strings alone do not explain the base model's lower MAUVE.

A post-run format audit of all 5,000 constrained outputs finds the standalone
uppercase word `OPTIONS` in 1,223 base outputs and four GRPO outputs, despite
zero occurrences of the forbidden literal `OPTIONS:`. Quiz/answer-style regex
markers occur in 1,805 base outputs, 96 GRPO outputs and 54 human controls.
These overlapping heuristic markers are not human quality labels. Manual
inspection of 20 specified examples finds grammatical sentences mixed with
quiz-format shifts, repetition and contradictions against the human reference.
The base outputs average 164.1 words, versus 224.0 for GRPO and 195.2 for human
controls. These distribution differences help explain low MAUVE, but the audit
does not isolate their causal contributions to the score. Marker definitions
and complete counts are in the corrected run's `format_audit.json`.

For SD1.5 diffusion evasion, the validation-selected checkpoint increases test
evasion from 1.2% to 8.2% while retaining 99.1% of mean CLIP alignment. A later
checkpoint reaches 99.4% evasion with only 63.3% alignment retention. The report
distinguishes the selected result from this exploratory high-evasion checkpoint.

### Watermark persistence and content preservation

- On 128 fresh questions, exposing TextSeal 2's estimated key-conditional
  probabilities reduces content-preserving direct-resampling removals from
  114/128 to 19/128. An adaptive ratio client still removes 126/128 watermarks,
  including 95 of its 96 outputs passing the quality rule.
- Support pruning reduces ratio-client joint removals from 21/32 to 3/32 or 1/32
  in a separate development comparison. Independent rewriting evades on all
  22 quality-preserving outputs. The 128-question experiment does not confirm
  this later pruning treatment.
- The image meta-distillation comparison shows no reliable defense gain.
  Under the revised preservation rule, the original model has 29/64 joint
  removals and each trained variant has 28/64. Paired uncertainty intervals
  include no difference. These counts are a post-hoc reanalysis.
- An independent 4,096-prefix human-null audit flags 5 prefixes at the unchanged
  nominal TextSeal 2 threshold of `p ≤ 0.001`. This diagnostic concerns the
  evaluated key and HC3 cohort.

Raw removal and content-preserving removal have separate denominators.
The report includes failed candidates, quality failures, paired uncertainty and
the distinction between development and confirmation. Automatic similarity
metrics provide proxies for content preservation.

## Translation attack study

The [Qwen3.5-9B instruct translation protocol](docs/experiments/translation.md)
warms the existing seven-detector ensemble, evaluates TextSeal 2 and the local
DistSeal-inspired text student with 5,000-sample MAUVE, then tests generation in
Simplified Chinese, French and Welsh followed by Index-Translate-9B or
TranslateGemma-12B translation into English. The same frozen ensemble scores
every condition. `scripts/translation_experiments.py` reuses the existing training
workers; `scripts/run_translation.slurm` checkpoints and requeues on one H100 in
development QoS. Full-paper reading and candidate defenses are recorded in the
protocol; defense implementation awaits stages I and II. No experiment results
are claimed before the run completes.
The split audit retained 984 detector training, 488 validation and 5,000 test
prompts. The warmed seven-member ensemble detects 99.76% of the unwatermarked
English test outputs, with 0.26% human false positives at threshold 0.5.
Job `92361` completed this baseline, then stopped on short watermark-training
responses. The revised watermark-only partitions retain 971 training and 485
validation responses of at least 128 tokens; all test data and detector results
remain unchanged. Cache cleanup freed approximately 6,500 file slots without
removing active dependencies or research artifacts. The obsolete portfolio-root
environment was then removed, releasing roughly 47,400 additional entries.
The corrected watermark inputs passed 25 focused CPU tests; job `96060` resumes
from the completed baseline. The TextSeal 2 English baseline is complete on 5,000 samples: ensemble detection
is 99.90% (0.26% human false positives), and watermark detection is 99.72%
(0.06% human false positives at the fixed p <= 0.001 threshold). Custom-feature
MAUVE averages approximately 0.0209 without the watermark and 0.0203 with it
across seeds 42, 43 and 44. These similarly low values do not alone establish
preserved quality. Translation-attack results remain pending. The user-requested two-H100 revision uses disjoint
DistSeal and non-DistSeal stage lists. TextSeal 2 MAUVE and clean/TextSeal 2
translation attacks can run while the latent student trains; locked result
merging preserves the shared cohort and feature caches. Jobs `107004`
(DistSeal) and `107002` (other work) run on separate H100 allocations; the
checkpoint handoff preserved student update 2,893. Following a hold at update
3,321, the user shortened DistSeal distillation to 5,000 total updates while
retaining its existing optimizer state and learning-rate schedule. Replacement
job `111368` resumes that branch; job `107002` is unchanged.

The current pilot is narrowed to English and Chinese-to-English only: the same
first 2,000 prompts, three model versions and two translators (12,000 translated
outputs). Jobs `116870`, `116871` and `116872` run one model branch per H100.
Generation/translation start at batches 32/16, with memory fallback and recorded
GPU telemetry. Existing texts, cached scores/features and the full 5,000-example
cohort are retained for extension. The revised pilot passed 37 focused CPU tests.

A separate French/Welsh pilot uses the first 200 held-out prompts per language
and model version, with both translators and the same frozen English-warmed
ensemble. Job `119154` requests one H100 in development QoS with checkpointing
and automatic requeue. Its 1,200 source texts and 2,400 translations are stored
under `french-welsh-200/`, alongside the English/Chinese pilot. It reuses the
existing watermarked models and calibration without training new detectors.
Job `119154` saved seven conditions before DivEye rejected one twelve-word
translation. Replacement job `122138` resumes the remaining work, explicitly
reporting unscorable short outputs while preserving all texts and the frozen
ensemble. The length-handling revision passed 18 focused tests.
The resumed job completed all twelve conditions. French ensemble detection is
97% to 100%; Welsh ranges from 82% to 98.49%, depending on model and translator.
TextSeal 2 watermark detection falls to 2% to 2.5% after French translation and
7% to 9% after Welsh translation. The local text student's source watermark is
already weak in these languages (4.5% French, 21.5% Welsh), so its translated
rates require comparison with those baselines. The report retains all counts,
including the one TextSeal 2/Welsh/Gemma output too short for DivEye.

A zero-shot generator-transfer pilot evaluates the same frozen Qwen-warmed
ensemble on 128 English continuations from `google/gemma-4-12B-it` and 128
CNN/DailyMail human controls. The paired first 128 held-out IDs have no overlap
with detector warmup or validation. Job `120562` uses one H100 in development
QoS, native instruction formatting with thinking disabled, and existing
checkpointed generation/scoring. The earlier base-model preparation was removed
at the user's request, freeing approximately 24 GB. The [transfer protocol](docs/experiments/generator_transfer.md)
records the command, model revision, validation and pilot limitations.
Job `120562` completed in 14m56s: at the unchanged ensemble threshold 0.5,
all 128 Gemma outputs were detected and 3/128 human controls were falsely
flagged (2.34%); ROC AUC was 1.0. This is a small news-domain transfer pilot.

## Code structure

Implementation modules live in `ai_content_detector/`, experiment entry points
and Slurm launchers in `scripts/`, and regression checks in
`ai_content_detector/tests/`. Generic ML utilities come from the pinned
`ml_pipeline` package maintained in [ML Playground](https://github.com/oremaz/ml-playground).

| Start here | What to inspect |
|---|---|
| [Detector contract](ai_content_detector/detectors/ensemble.py) | AI score direction, uncertainty, aggregation and resource release |
| [Text optimization](ai_content_detector/rl_evasion/text_evasion/grpo_optimization.py) | TRL updates, reference state and explicit fallback |
| [Diffusion policy](ai_content_detector/rl_evasion/image_evasion/diffusion_policy.py) | Transition log densities, reference KL and saved trajectories |
| [Paired defender evaluation](ai_content_detector/rl_evasion/defender_comparison/workflow.py) | Saved-text reuse and resumable experiment phases |
| [Watermark preservation](ai_content_detector/watermarking/preservation.py) | Input alignment, blind quality selection and joint outcomes |
| [Support optimization](ai_content_detector/watermarking/exposure_support.py) | Evidence floors and retained-mass constraints |
| [Watermark experiment driver](scripts/instruct_watermark_experiments.py) | Frozen manifests, disjoint questions and staged owner evaluation |

The [evasion guide](ai_content_detector/rl_evasion/README.md) and
[watermarking guide](ai_content_detector/watermarking/README.md) give the module
maps, experiment commands and resource requirements.

## Setup and validation

Use Python 3.11+ and run from the repository root.

```bash
uv venv
uv pip install -r requirements.txt
uv run --no-sync python -m spacy download en_core_web_sm
```

Validate the portable results without loading model weights or local datasets.

```bash
PYTHONPATH=. uv run --no-sync python scripts/publish_watermark_results.py --check
```

Run the CPU test suite with bounded numerical-library threads.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=. uv run --no-sync python -m pytest ai_content_detector/tests -q
```

The tests cover score semantics, formulas, splits, artifact integrity, failure
behavior and restart contracts. Real-model inference, GPU training and external
APIs require their recorded resources. Historical test receipts describe their
original checkouts. The [validation record](tex/records/docs/cleanup_validation.tex)
documents environment-specific CPU and notebook-kernel checks.

On 2026-11-11, the current checkout passed 538 CPU tests with one notebook
signal test deselected because the sandbox denies local TCP binding. This run
used writable temporary caches and disabled optional GPU-only kernels in the
test process to exercise the CPU reference implementation. GPU training and
real-model inference were not rerun.

The smoke experiment checks CPU wiring. The application needs the detector
dependencies and weights described in the catalogue.

```bash
PYTHONPATH=. uv run --no-sync python -m ai_content_detector.rl_evasion.run_experiments \
  --experiment smoke_test
PYTHONPATH=. uv run --no-sync streamlit run ai_content_detector/app.py
```

## Reproducibility and report sources

Prompt splits are deterministic and fixed before training. MAML separates support
and query prompts, with both contributing to training. Benchmark manifests check
row alignment and declared held-out detector overlap. Attack checkpoints do not
yet record training sample IDs, so sample-level overlap cannot be rejected
automatically.

Slurm workflows retain manifests, frozen code, input hashes and resumable state.
Reproduce a recorded run using its original snapshot and artifacts. Model weights,
datasets, runtime caches and raw job logs are local resources excluded from Git.
Portable metrics and fixed-sample galleries are retained with the documentation.

The [research guide](tex/records/docs/research_guide.tex) contains the historical
setup and experiment commands. [TeX records](tex/records/) preserve detailed
protocols, and the [migration map](tex/provenance/migration.json) links them to
hashed source snapshots. The [compute ledger](tex/provenance/compute_budget.json)
records the development experiments' 60 H100-hour budget and historical usage.

Build the modular report with XeLaTeX and BibTeX.
On the research cluster, first load `texlive/20240312`. The Gemma and translation
tables regenerate from the checked-in portable metrics without GPU inference:

```bash
PYTHONPATH=. uv run --no-sync python tex/translation_summary.py
```

```bash
cd tex
latexmk -xelatex report.tex
```

The revised PDF has 48 pages, including figures, tables and 1 reference page.
The text-only build has 29 pages excluding references, within the roughly
40-page text limit. The [layout audit](tex/provenance/layout_audit.json) records
the current PDF hash, page-count method and checks. This translation/report
revision passed 21 focused CPU tests and an independent cache audit across 22
evaluation conditions. The earlier watermark publication audit remains recorded
separately in the layout audit's historical validation.

The report includes paired detector-evasion and watermark-removal examples,
mathematical attack descriptions, and explicit target-access labels. The main
image-removal table applies the retained LPIPS, image-CLIP and class-retention
rule to all saved pairs; historical SSIM-gated results remain separate.
[The image audit](tex/image_preservation_audit.py) and
[qualitative exporter](tex/qualitative_evidence.py) regenerate these assets with
`PYTHONPATH=. uv run python tex/image_preservation_audit.py` and
`PYTHONPATH=. uv run python tex/qualitative_evidence.py`. Regeneration needs the
frozen local experiment artifacts; building the checked-in TeX and PDF assets
does not require model inference. The qualitative provenance receipt records
source hashes and fixed example positions.
