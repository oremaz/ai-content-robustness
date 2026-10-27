# Cleanup validation (2026-10-27)

The cleanup separates text and image detectors into method-specific modules.
The original detector import modules and serialized class identities remain
available. The long README is retained as the [research guide](research_guide.md),
with its links resolved from the new location.

## Checks in this checkout

- Broad research-environment run: **451 passed**, two environment-sensitive
  failures, one notebook integration check excluded.
- HC3 JSON loading passed separately with a writable dataset cache selected
  before the `datasets` package was imported.
- The Qwen3.5 second-order CPU test passed separately using Transformers' native
  PyTorch reference functions. The optional CUDA convolution and FLA imports
  were blocked in that test process only. No installed packages were changed.
- The notebook checkpoint-signal test passed separately with local Jupyter
  sockets available outside the filesystem/network sandbox, under a 180-second
  process timeout.
- Detector import/serialization and image normalization checks passed.
- The CLI smoke test reported every component OK. The README's synthetic-image
  example ran. Streamlit startup passed with detector availability mocked out;
  this did not scan or load real detector weights.
- A detector serialized using the original source reloaded through the new
  imports with identical state and score-calibration outputs. Syntax-tree
  comparisons preserved all 19 original detector class/function definitions,
  apart from historical module identities and documentation wording.

Thus all **454 collected checks passed across the broad and isolated runs**,
with the environment qualifications above. This is not a claim that an
unmodified full-suite command passes in the sandbox.

## Commands

Run from this repository's root, using its existing `.venv` without synchronizing
or upgrading the experiment dependencies:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2
PYTHONPATH=. uv run --no-sync python -m pytest ai_content_detector/tests \
  -q --tb=short -p no:cacheprovider -k 'not papermill_forwards_scheduler_signal'
```

The JSON cache check can be repeated with a writable temporary cache:

```bash
PYTHONPATH=. HF_DATASETS_CACHE=/tmp/ai-robustness-validation-cache \
  uv run --no-sync python -m pytest \
  ai_content_detector/tests/test_benchmark_contracts.py::test_hc3_json_loader_preserves_order_answers_and_source_revision \
  -q -p no:cacheprovider
```

The isolated CPU reference-kernel check used:

```bash
PYTHONPATH=. uv run --no-sync python - <<'PYTEST'
import sys
sys.modules["causal_conv1d"] = None
sys.modules["fla"] = None
import pytest
raise SystemExit(pytest.main([
    "ai_content_detector/tests/test_meta_restart.py::test_native_causal_model_supports_full_second_order_meta_step[qwen35]",
    "-q", "-p", "no:cacheprovider",
]))
PYTEST
```

Run the notebook signal test where local sockets are permitted:

```bash
PYTHONPATH=. timeout --signal=INT --kill-after=10s 180s \
  uv run --no-sync python -m pytest \
  ai_content_detector/tests/test_training_restart.py::test_papermill_forwards_scheduler_signal_to_checkpointing_kernel \
  -q -p no:cacheprovider
```

## Experiment preservation

No training algorithms, requirements, model/data artifacts, job launchers,
configuration defaults or frozen source snapshots were changed. Read-only Slurm
checks before and after the cleanup accounted for the same 17 pending jobs.
Full GPU experiments, detector-weight loading and live APIs were not rerun.
Reproduce a recorded experiment using its original manifest, frozen source,
environment, data, checkpoints and random-state receipts.

## Component-guide follow-up

Added task/protocol and code maps for RL evasion and watermarking. After the
shared ML Playground pipeline extraction, **24 focused compatibility/comparison
checks passed** against its current sibling source:

```bash
PYTHONPATH=.:../ml-playground uv run --no-sync python -m pytest \
  ai_content_detector/tests/test_detector_import_compatibility.py \
  ai_content_detector/tests/test_defender_comparison.py -q -p no:cacheprovider
```

This follow-up did not rerun the broad suite or full GPU experiments. A final
read-only queue check still showed the same 17 pending jobs.

## Training-code follow-up after the request audit

The GRPO entry point was reduced from 698 to 327 lines, and the DDPO entry point
from 1,020 to 389 lines. Optimization, diffusion-policy mathematics, image rewards
and model/evaluation lifecycle are separate modules linked from the package guide.
These are targeted extractions; other research files were not rewritten solely
to meet a line-count limit.

All 52 original function/method definitions retain their executable syntax
trees. Documentation was clarified, including the distinction between the
REINFORCE fallback's squared log-ratio approximation and TRL GRPO. Original
trainer import paths and serialized prompt-statistics class identities remain.
No loss, generation, seed, split or restart-update statements were changed.

The final focused run passed **71 checks, with one notebook signal check
excluded**. It covers DDIM densities, reference KL, reward components, prompt
groups, learning-rate schedules, interruption/resumption, evaluation recovery,
TRL callbacks and historical imports/serialization. An initial run caught a
missing `json` import in the extracted optimization module; the import was
restored and all 71 checks passed on the final run. The broad 454-check suite was
not rerun for this follow-up.

```bash
PYTHONPATH=.:../ml-playground uv run --no-sync python -m pytest \
  ai_content_detector/tests/test_ddpo_math.py \
  ai_content_detector/tests/test_grpo_contracts.py \
  ai_content_detector/tests/test_grpo_math.py \
  ai_content_detector/tests/test_published_image.py \
  ai_content_detector/tests/test_reference_reward.py \
  ai_content_detector/tests/test_image_attacker_ablation.py \
  ai_content_detector/tests/test_attacker_schedule.py \
  ai_content_detector/tests/test_evaluation_restart.py \
  ai_content_detector/tests/test_training_restart.py \
  ai_content_detector/tests/test_reward_optimization.py \
  ai_content_detector/tests/test_trainer_import_compatibility.py \
  -q -p no:cacheprovider -k 'not papermill_forwards_scheduler_signal'
```

A read-only queue check after the code changes again showed the same 17 pending
jobs. Frozen source directories and submitted job commands were not changed.
