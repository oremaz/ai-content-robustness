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

```bash
cd tex
latexmk -xelatex report.tex
```

The revised PDF has 33 pages. The [layout audit](tex/provenance/layout_audit.json)
records the previous edition's PDF hash and page-count method.
