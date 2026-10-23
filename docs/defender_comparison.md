# Defender updates from retained attackers

## Current scope

Three independent jobs use the retained Qwen3.5-9B-Base attackers. Each updates its warmed defender(s) once, then evaluates. No attacker optimizer runs, and no second round starts.

| Branch | Retained attacker updates | Defender(s) | Job |
| --- | ---: | --- | ---: |
| gemma | 1213 | decoder_classifier | 1447844 |
| modernbert | 1487 | modernbert_large | 1447845 |
| ensemble | 502 | binoculars, fast_detect_gpt, diveye, roberta, tfidf, modernbert_large, decoder_classifier | 1447846 |

The notebook decoder is `qlora_gemma4_e4b`; the encoder is `qlora_modernbert_large`. The ensemble contains Binoculars, Fast-DetectGPT, DivEye, RoBERTa, TF-IDF, ModernBERT-large and the decoder classifier. All begin from their completed Qwen warm-up checkpoints.

## Completed defender updates

All three jobs completed successfully without requeue, within their two-hour
development allocations. On the same 500 saved attack texts and 500 human
passages, at threshold 0.5:

| Defender | Evasion before | Evasion after | Human false positives after | Allocation time |
| --- | ---: | ---: | ---: | --- |
| Gemma | 77.2% | 0.2% | 0.2% | 47m35s |
| ModernBERT-large | 96.8% | 1.0% | 0.4% | 41m45s |
| Seven-detector ensemble | 94.8% | 0.4% | 0.0% | 1h31m13s |

The updated ensemble members differ: DivEye retains 7.0% evasion with 5.8%
human false positives, and TF-IDF retains 5.8% evasion with 7.0% human false
positives. The aggregate ensemble score should therefore be read alongside the
per-member tables in the results report. No attacker was trained further.

## Retained evidence

The [results report](research_results.md) retains only the step-500/final Gemma and ModernBERT models, the step-500/502 ensemble models, and the separate RoBERTa step-500 model and fluctuation measurements. The ensemble full evaluation was measured at 502. The individual long runs have step-500 snapshots but no separate full detection measurement exactly at 500. Do not substitute training reward for detection performance.

## Training and evaluation pairs

Each branch generates once on the existing 1,000 fitting prompts from CNN/DailyMail. The existing eligibility rules retain aligned AI/human pairs with at least 20 words in both classes. Thus there are at most 2,000 labelled examples. Torch defenders retain their existing three-epoch supervised update at learning rate 2e-5, batch size eight and maximum length 512. The paired 80/20 split keeps each prompt and its AI/human versions together. Feature classifiers retain their existing architecture-specific update, including continued XGBoost training for DivEye.

The 500 reserved evaluation prompts never enter defender fitting. Initial metrics reuse the completed saved evaluations after checking checkpoint, warmed-defender and dataset hashes. After defense, the same saved generated texts and human passages are scored with the updated defender(s). The semantic similarity therefore remains a property of those fixed attack texts. The ensemble reports each member and the arithmetic mean of all seven AI scores.

## Checkpointing and allocation

All three request development H100 QoS and two hours per allocation. Gemma and ModernBERT request 32 physical CPUs; the ensemble requests 48 for host memory while retaining one GPU. The previous reported supervised-fit runtimes ranged from 22.67 to 750.4 seconds, with the completed Gemma fit reporting 396.1 seconds. Generating 1,000 training outputs and scoring the updated defenders add time, so these observations are not a runtime guarantee.

Generation saves each batch of eight. Supervised training saves native model, optimizer, scheduler and RNG state, with a five-minute checkpoint clock and a ten-minute Slurm warning. The phase journal saves each completed defender separately. A pause requeues the same job; completed data generation, defender updates and measurements are not repeated. Each branch writes to its own directory, including the individual Gemma/ModernBERT updates and their counterparts inside the ensemble.

## Running and reading the jobs

```bash
PYTHONPATH=. uv --no-cache run --no-sync python scripts/retained_races.py prepare
PYTHONPATH=. uv --no-cache run --no-sync python scripts/retained_races.py submit NEW_MANIFEST
PYTHONPATH=. uv --no-cache run --no-sync python scripts/architecture_races.py release NEW_MANIFEST
```

Submission holds the three jobs until resource and implementation checks complete. Each manifest records the retained source hashes, cached initial measurements and exact source text pairs. Shared metrics are locked and idempotent. The report exports tables, paired before/after curves, defender training losses, PNG/SVG figures and their source CSVs.

## Base-model retention check

After the completed update, all three branches are scored on the original
500 reserved base-Qwen outputs and their 500 human passages. These are the
saved base outputs, rather than outputs from an RL checkpoint. No model is
trained or used to generate text in this check. Each updated checkpoint and
the evaluation inputs are verified against recorded hashes before scoring.
The comparison uses the original warmed-detector scores and threshold 0.5.
All seven ensemble members and their arithmetic mean are reported separately.

```bash
PYTHONPATH=. uv --no-cache run --no-sync python scripts/evaluate_updated_base.py prepare COMPLETED_MANIFEST
PYTHONPATH=. uv --no-cache run --no-sync python scripts/evaluate_updated_base.py submit BASE_MANIFEST
PYTHONPATH=. uv --no-cache run --no-sync python scripts/evaluate_updated_base.py release BASE_MANIFEST
```

The three independent, one-H100 development jobs use two-hour allocations.
Scoring saves every eight texts and resumes without rescoring completed
batches. Results and before/after figures are added to the same curated report,
with a separate shared base-evaluation metrics file.

Submitted evaluations: Gemma `1460317`, ModernBERT `1460318`, and ensemble
`1460319`. The input and updated-checkpoint hashes were verified before
release. The base-evaluation, submission and reporting checks passed 26
focused tests in 417.35 seconds, including the shared-environment import time.

All three completed with exit code zero and no restart: Gemma in 2m54s,
ModernBERT in 1m40s and the ensemble in 5m50s.

| Updated branch | Base AI detection before | Base AI detection after | Human false positives after | AUROC after |
| --- | ---: | ---: | ---: | ---: |
| Gemma | 100.0% | 100.0% | 0.2% | 0.999992 |
| ModernBERT-large | 99.0% | 90.2% | 0.4% | 0.997226 |
| Seven-detector ensemble | 100.0% | 98.4% | 0.0% | 0.999976 |

The ensemble's TF-IDF member falls from 88.6% to 28.2% base detection, with
7.0% human false positives and AUROC 0.615786. Thus recovery on RL-generated
texts can accompany a substantial loss on base outputs for an individual
member. ModernBERT also loses 8.8 percentage points of base detection in its
individual branch, while its human false positives fall from 1.6% to 0.4%.
The full per-member comparison and figures are in the results report.

## Interpretation limits

The long-run attackers use different historical training budgets and targets. This experiment measures defender recovery against those fixed attackers; it does not establish robustness to a newly retrained attacker or asymptotic robustness. The original diagnostic set has already informed protocol choices. Semantic similarity does not guarantee factual accuracy, and no KL-divergence measurement was recorded for the old no-KL runs.
