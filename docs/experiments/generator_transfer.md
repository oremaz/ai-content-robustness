# Zero-shot Gemma generator transfer

The pilot tests the frozen seven-member ensemble warmed on Qwen outputs against
128 English outputs from [Gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it),
pinned at revision `43c9d014b231474fa7212dbb48a762566c2b255d`. Detectors receive
no Gemma-12B adaptation, retraining or new threshold calibration.

The inputs are the first 128 test IDs from the translation study's fixed
5,000-example CNN/DailyMail 3.0.0 test cohort. Human controls are the matching
saved article continuations, capped at 256 Qwen tokens as in the main study.
ID and human-reference checks find zero overlap with detector warmup and
validation. The main 2,000-sample translation cohort is likewise disjoint from
those adaptation and validation IDs.

Gemma receives the same English news-continuation instruction as Qwen, using
its native chat template with `enable_thinking=False`. Prompt construction
checks token-for-token equivalence with the native formatted template to avoid
duplicating or omitting the BOS token. Generation uses 256 new tokens, batch 32,
temperature 1.0, top-p 0.95, the model's top-k 64 and seed 42. The existing
`generate_batched` helper retains generated batches and RNG state for resuming.

`scripts/evaluate_generator_transfer.py` reuses the existing defender pool,
checkpointed scoring, ensemble mean and classification metrics. It reports
detection and human false positives at threshold 0.5, plus the unchanged
Qwen-validation threshold targeting 1% false positives. Per-member scores,
short/empty-output counts and ROC AUC are diagnostic results. No watermark or
MAUVE work is part of this side experiment.

The translation Slurm launcher accepts an optional third argument naming a
frozen worker script, preserving its existing default behavior:

```bash
sbatch scripts/run_translation.slurm MANIFEST STAGES_JSON FROZEN_WORKER
```

Run artifacts live in
`results/research_workflow/generator_transfer/gemma4-12b-it-128-20261112T223454Z/`.
Job `120562` requests one H100 in development QoS for up to two hours, with
checkpointing and automatic requeue. Its immutable inputs are `manifest-v2.json`,
`stages.json` and `source-v2/scripts/evaluate_generator_transfer.py`. Results
are written to `results.json`; the saved predictions retain each cohort ID.

Validation checked all 128 native prompts, complete pinned weights, all 1,674
frozen hashes and cohort separation. Seventeen focused transfer/translation
tests passed, and the launcher passed its shell syntax check. Initial job
`120510` stopped before generation on the special-token guard; version two
corrected BOS handling before resubmission. The failed manifest and log are
retained for provenance. The unsubmitted base-model run, its model cache and
unreferenced base-only shared blobs were deleted at the user's request,
releasing approximately 24 GB while preserving all instruct files.

This is a 128-sample pilot of generator transfer within the news domain. Each
sample changes a class-specific rate by 0.78125 percentage points. Disjointness
from local detector adaptation does not establish absence of these articles
from the original pretrained models' data. No result is claimed before scoring
completes.
