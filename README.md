# AI Content Robustness

A research workbench for measuring how AI-content detectors respond to adaptive
text and image generators, and whether retraining recovers detection. It also
contains watermarking and removal experiments with separate attacker and owner
roles. A Streamlit app exposes the detector interface for interactive scoring.

The central experiment is a controlled loop: train a generator against a detector,
evaluate on reserved prompts, retrain the detector on fresh training outputs,
and score the same saved evaluation texts again. Scores measure statistical
patterns, not proof of authorship.

## Results and scope

The [retained results](docs/research_results.md) evaluate Qwen3.5-9B-Base LoRA
attackers on 500 generated texts and 500 human passages. Evasion is the fraction
of generated texts with AI score below 0.5:

| Target | Attacker updates | Evasion before defender update | After one defender update |
|---|---:|---:|---:|
| Gemma | 1,213 | 77.2% | 0.2% |
| ModernBERT-large | 1,487 | 96.8% | 1.0% |
| Seven-detector ensemble | 502 | 94.8% | 0.4% |

Before and after use identical saved texts. Training budgets differ, so the rows
are not a matched-budget comparison. The experiment shows recovery against fixed
attackers; another adaptive attacker round remains untested. Base-generator
retention and individual ensemble members expose additional tradeoffs in the
report. Semantic similarity does not certify factual accuracy.

Watermarking is a separate line of work. See the [protocol](docs/watermarking.md)
and [removal review](docs/watermark_removal_literature.md) for TextSeal, released
DistSeal image checkpoints, and the experimental latent text adaptation.
Pending jobs are not completed results.

## Read the code

| Component | Start here | What to inspect |
|---|---|---|
| Detector contract | [ensemble.py](ai_content_detector/detectors/ensemble.py) | AI score direction, uncertainty, aggregation and resource release |
| Text statistics | [likelihood_text.py](ai_content_detector/detectors/likelihood_text.py) | Binoculars and Fast-DetectGPT score calculations |
| Image detector | [warpad.py](ai_content_detector/detectors/warpad.py) | Feature extraction, raw statistics and threshold semantics |
| Text training lifecycle | [grpo_trainer.py](ai_content_detector/rl_evasion/text_evasion/grpo_trainer.py) | Setup, explicit backend selection and evaluation |
| Text optimization | [grpo_optimization.py](ai_content_detector/rl_evasion/text_evasion/grpo_optimization.py) | TRL updates, reference state and the explicit fallback |
| Diffusion mathematics | [diffusion_policy.py](ai_content_detector/rl_evasion/image_evasion/diffusion_policy.py) | Transition log probabilities, reference KL and trajectories |
| Paired defender evaluation | [workflow.py](ai_content_detector/rl_evasion/defender_comparison/workflow.py) | Saved-text reuse and resumable experiment phases |
| Watermark removal | [watermark_removal.py](scripts/watermark_removal.py) | Attacker contracts, source snapshots and owner evaluation jobs |

The [evasion package guide](ai_content_detector/rl_evasion/README.md) and
[watermarking package guide](ai_content_detector/watermarking/README.md) explain
which workflow to inspect and where its evaluation boundaries sit.

For a short first pass, read the detector contract and paired evaluation workflow,
then inspect one detector. The [detector catalogue](DETECTORS.md) distinguishes
published implementations from local heuristics. The [full research guide](docs/research_guide.md)
retains method explanations, equations, examples, commands and historical validation records.

## Run

Use Python 3.11+ and run from this repository's root:

```bash
uv venv
uv pip install -r requirements.txt
uv run python -m spacy download en_core_web_sm
PYTHONPATH=. uv run python -m ai_content_detector.rl_evasion.run_experiments \
  --experiment smoke_test
PYTHONPATH=. uv run streamlit run ai_content_detector/app.py
```

The smoke test checks CPU wiring, not real-model accuracy. Most detectors need
external weights or fitted checkpoints. Training needs the appropriate GPU stack.
The applications install the shared `ml_pipeline` package from a pinned Git commit
in [ML Playground](https://github.com/oremaz/ml-playground).

Use [GPU setup](docs/research_guide.md#1-create-the-environment) for the existing
CUDA and Qwen kernel pins, [resource requirements](docs/research_guide.md#required-models-and-external-resources)
for weights and authentication, and [experiment commands](docs/research_guide.md#running-experiments)
for GRPO, SPIN, DDPO, MAML and benchmarks. Watermarking has its own optional
requirements and Slurm commands in the protocol above.

A lightweight detector example needs no weights:

```python
from PIL import Image
from ai_content_detector.detectors import FrequencyDetector

result = FrequencyDetector().detect(Image.new("RGB", (256, 256), "gray"))
print(result.score, result.label, result.details)
```

This is an interface example using a synthetic image, not a detection benchmark.

## Evaluation and reproducibility

Prompt splits are deterministic and fixed before training. MAML separates support
and query prompts; query prompts contribute to training. Benchmark manifests
check row alignment and reject declared held-out detector overlap. Attack
checkpoints do not yet record training sample IDs, so sample-level overlap cannot
be rejected automatically. See [evaluation controls](docs/research_guide.md#evaluation-and-leakage-controls).

Existing CLI names, configuration defaults, seeds and checkpoint paths are
preserved. `text_detectors.py` and `image_detectors.py` retain their public imports;
implementations live in smaller method-specific modules. Serialized detector
classes retain their historical module identities.

Slurm workflows use saved manifests and frozen source directories. Resume the
original run using its original manifest, snapshot, environment and artifacts.
Do not substitute a new run's configuration when reproducing a recorded result.
The [defender protocol](docs/defender_comparison.md) documents retained checkpoints
and continuation; the watermarking protocol documents its source snapshots.

## Tests

```bash
PYTHONPATH=. uv run python -m pytest ai_content_detector/tests -q
```

CPU fixtures and mocks check score semantics, formulas, splits, failure behavior
and restart contracts. GPU training, external APIs and real-model inference need
their recorded resources. Historical test receipts in the guide describe their
original checkouts rather than certifying the current environment.

The [2026-10-27 cleanup checks](docs/cleanup_validation.md) record the current
validation results, CPU/kernel workarounds and pending-job preservation checks.
