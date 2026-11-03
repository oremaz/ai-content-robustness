# BIRA replication and Qwen extensions

The replacement run uses the released BIRA implementation at revision
`6f62ecce6f3410514fd43a40583a4059822af73a`. The full paper, source,
model revisions and environment receipt are archived under
`results/research_workflow/watermarking/resources/`.
The previous adapted BIRA outputs and metrics were deleted; probability
resampling and smoothing results were retained.

## Released local-model experiment

The control uses the authors' standard Meta-Llama-3.1-8B-Instruct checkpoint,
without attack-specific fine-tuning. The driver executes unchanged released
class definitions, omitting unrelated API imports. It preserves the system
prompt, document framing, causal surprisal calculation, median token selection,
beta -4, temperature 0.7, top-p 0.95, top-k 50, 1,500-token limit, seed reset on
every generation, decoded-text repetition check and released 20-attempt retry
schedule with beta increments of 0.125. Ordinary paraphrasing is a separate arm.

Seven released watermark schemes use the first 500 released C4 examples and
OPT-1.3B. This covers the local 8B arms, not the paper's 70B, API, DIPPER, SIRA
or paid GPT evaluations. Generation receives explicit per-example seeds for
restart reproducibility. Released metrics are saved separately from metrics
using disjoint null-only validation calibration and correct score directions,
including EXP's lower-is-watermarked p-value.

## TextSeal and stronger latent Qwen

TextSeal receives both the Llama control and a Qwen3.5-9B extension. Qwen uses
bfloat16, its native chat template with thinking disabled, and a separate
50-document C4 beta calibration, disjoint from test and threshold validation.
Its adaptation and calibration are recorded explicitly.

The stronger DistSeal-inspired Qwen student has ordinary-sampling detection
160/256 (62.5%); its teacher has 177/256 (69.1%). This is the repository's
experimental latent text adaptation, not an official DistSeal text checkpoint.
The attack runner applies BIRA, direct top-k probability resampling and
confidence-based smoothing, plus an ordinary paraphrasing control. The old
20-example naturalness gate has been removed from scheduling and execution
after the matched unwatermarked base also failed it. No naturalness score is
required to run the stronger student. Historical diagnostics remain recorded,
and `--quality-only` runs them independently without affecting attacks.
An initial one-sentence-prefix diagnostic used a different context length and
is superseded; later diagnostics used the original 50-token prefixes.

The probability attacks use the student's own next-token distribution without
installing the teacher's watermark hook. Owner detection is evaluated after
attack artifacts are frozen, using the original stronger-student threshold.
Their temperature and nucleus cutoff are read from the student's original
frozen `generation_config`: temperature 0.8 and top-p 1.0 for this checkpoint.
The historical naturalness probes instead used top-p 0.95. They were matched
across the base and students, but did not exactly reproduce the original
sampling protocol and are not suitable as an eligibility check. The released
BIRA rewriter retains its separate temperature 0.7 and top-p 0.95 settings.

## Quality and execution

Rewrites retain official CEnli, MPNet, SelfBLEU and Mistral perplexity metrics,
plus reverse entailment and repetition, truncation, numeric-change and
word-gluing diagnostics. Detection evasion is reported separately from quality.
Fresh probability-generated continuations are not compared semantically to an
unrelated previously sampled continuation.

From the project root:

```bash
PYTHONPATH=. uv run python scripts/bira_replication.py prepare
PYTHONPATH=. uv run python scripts/bira_replication.py submit /absolute/path/to/manifest.json
PYTHONPATH=. uv run python -m pytest ai_content_detector/tests/test_bira_replication.py
```

To test the earlier student (122/256, 47.7% ordinary-sampling detection), without
rerunning the paper or TextSeal cases:

```bash
PYTHONPATH=. uv run python scripts/bira_replication.py prepare --latent-only \
  --latent-manifest results/research_workflow/watermarking/20261026T100809190155Z/manifest.json \
  --reuse-calibration results/research_workflow/bira_replication/20261103T014241041156Z/manifest.json
```

Submit the resulting manifest with the command above. The checkpoint's own
extractor and frozen detection threshold are used. The 20 validation groups,
50-token prompts and attack settings stay fixed. Naturalness diagnostics do
not block attack execution. Verified reuse of beta calibration avoids a
separate calibration job in student-only runs.
Calibration reuse checks the clean Qwen rewriter, disjoint C4 documents,
released source, calibration driver, seed and sampling configuration, and
records the original result hash. It does not reuse the stronger student's
quality judgments or detection threshold.

The earlier-student run `20261103T090630596895Z` completed its 20-example
quality check: only 3/20 passed the automated gate, with 18 valid judge
responses and six explicit control-token/corruption flags. Manual inspection
confirmed repetition loops and unexpected quiz/assistant text; some automated
passes also contain quiz formatting, so 3/20 is not a count of certified usable
outputs. The conditional removal job `1888242` was cancelled after this failed
gate, before attack execution. No removal efficacy results exist for this run.
Eight focused driver tests passed, including calibration-reuse drift checks,
failed-gate blocking and submission with an expired Slurm preflight dependency.

The unwatermarked base can be checked with the identical validation examples,
seeds, 50-token prefixes, sampler and judge using:

```bash
PYTHONPATH=. uv run python scripts/bira_replication.py prepare --quality-only --quality-model base \
  --latent-manifest results/research_workflow/watermarking/20261026T100809190155Z/manifest.json
```

Submit the resulting manifest normally. This selects the original frozen
Qwen3.5-4B-Base weights and submits only the naturalness phase, without beta
calibration, attacks or detector evaluation. Failure of a student alone does
not establish whether the watermark training caused the quality problems;
the matched base control is needed for that comparison.

The matched base-control run is `20261103T093437669196Z`, job `1890474`.
Its `paired_control.json` records that all 20 validation IDs and the sampling
configuration match the earlier-student check. Ten focused driver tests passed,
including original-weight selection and submission of a quality-only job
without removal attacks.

New standalone quality probes read the same frozen original sampling recipe
as the probability attacks, including top-p 1.0 for the stronger checkpoint.
Use `--quality-model teacher` with `--quality-only` to check the external
watermarked teacher: this loads the original base plus the saved selected
embedder and message. `--quality-model student` loads the merged stronger
student without installing a teacher hook. All use the same first 20 unique
validation groups and per-example seeds. The teacher hook is removed before
the clean judge runs. This comparison remains diagnostic and does not block
the removal experiment.

The corrected standalone probes are teacher job `1896834`, stronger-student
job `1896837` and base-control job `1896839`. They generate 20 validation
continuations each at temperature 0.8 and top-p 1.0. Teacher and student may
run concurrently; base waits for the teacher to finish, keeping the total
within four GPUs alongside existing work. Fourteen focused driver tests passed,
including exclusive teacher-hook installation and cleanup on interruption.

While these probes wait for shared capacity, `saved_20_sample_audit.json` in
run `20261103T112623223366Z` records inspection of the first 20 fixed test
outputs per teacher/student from the original top-p 1.0 evaluation. Both have
7/20 quiz/exercise-format insertions; teacher has 2/20 literal reasoning/control
markers and student has 3/20. Repetition and incoherent passages remain visible.
These are descriptive flags with per-sample notes, not a usability accuracy.
Repeated option templates can trigger the repetition proxy. This saved-test
audit does not quantify improvement against the different validation examples
in the old top-p 0.95 probes; the fresh paired probes are needed for that.

The base control completed successfully and failed the quality gate: 6/20
continuations passed, versus 3/20 for each student on the same validation IDs.
The base had 19 valid judge responses and six explicit corruption/control-token
flags. It also produced repetition and unexpected quiz/assistant text.
`quality_comparison.json` records all three reports and their hashes. These
small, automated comparisons do not establish the size or cause of a quality
difference, and some judge passes contain quiz formatting. In particular, the
student failures cannot be attributed solely to watermark training because the
original base already fails under this raw-continuation protocol. No removal
attacks were run in the base control.

The stronger-student removal run is `20261103T104924684074Z`, job `1895834`.
It runs all three methods and the paraphrasing control on the original 256
stronger-student test examples, without the naturalness prerequisite. The
probability oracle uses the frozen original temperature 0.8 and top-p 1.0;
the earlier preparation `20261103T104335074066Z` was cancelled before attack
outputs were saved to correct its top-p setting. The original 62.5% baseline,
extractor and owner threshold remain fixed. Fourteen focused tests passed,
covering nucleus-distribution equivalence at both top-p settings, hybrid-cache
consistency, original-recipe loading and execution despite a failed standalone
quality diagnostic. `sampler_audit.json` records the audit and its limits.

Preparation requires the frozen models and completed isolated Python 3.10 paper
runtime archive. Jobs extract that runtime node-locally; Qwen uses the project
environment. Compiler caches also stay node-local and are removed at job exit
to avoid exhausting the shared filesystem's file-count quota. GPU preflight exercises both rewriters and all seven released
watermark generators before dependent work. Jobs use two-hour H100 dev QoS,
checkpoint completed examples and rewrite attempts, and requeue on the warning
signal. A generation interrupted inside an unfinished attempt can require
recomputation; zero compute loss is not guaranteed. Model reload and runtime
extraction also have overhead.

Two paper workers use at most two GPUs, with the Qwen/TextSeal
branch limited to one. The latter runs calibration, then TextSeal Llama,
TextSeal Qwen and the latent-Qwen case in sequence. Student-only runs do not
resubmit the paper or TextSeal cases.
All phases, failures, frozen hashes and job IDs are retained in the run manifest
and phase receipts. A submitted or pending job is not a completed result.

Validated in this checkout: 13 focused CPU tests passed across the BIRA driver
and probability-attack helpers, including released source
execution, the 20-retry schedule, causal surprisal, seed reset, attempt replay
and the strict naturalness gate. The submitted run
`20261103T012950108624Z` also passed both Llama and Qwen preflights on an
80 GB H100, including native generation and detection for all seven paper
schemes. These smoke checks do not establish removal efficacy or full-run
quality; those require the experiment to complete.

The corrected 50-token-prefix run is `20261103T014241041156Z`. Its
`reuse_receipt.json` records the verification and reuse of both GPU preflights
and 223 completed paper generations from the superseded run. No initial
short-context naturalness judgments are reused. The quality-check job is
`1878623`, followed by extension worker `1878624`; paper workers are `1878620`
and `1878621`. Paper workers wait until the quality-check worker has started
so it receives the first available GPU slot. Shared QoS capacity can still
delay execution.

## Completed stronger-student results

All three corrected quality probes and stronger-student removal job `1895834`
completed with exit code 0. On the same 20 validation examples at temperature
0.8 and top-p 1.0, the automated diagnostic qualified 8/20 base continuations,
2/20 teacher continuations and 6/20 stronger-student continuations. Valid judge
responses were 18/20, 19/20 and 18/20 respectively. Mean fluency/coherence
among valid judgments was 3.67/3.33 for base, 2.63/2.21 for teacher and
2.67/2.22 for student. The student's count increased from 3/20 at top-p 0.95,
but mean scores changed only slightly and quality problems remain. This is a
small automated diagnostic, not a usability accuracy or an execution gate.
`corrected_quality_comparison.json` records the comparison and actual
per-example seeds (0 plus validation index).

Frozen-owner detection on all 256 stronger-student test outputs:

| Condition | Detected | Detection rate | Eligible for the 32-token extractor minimum |
|---|---:|---:|---:|
| Original student | 160/256 | 62.5% | 255/256 |
| Direct top-k resampling | 153/256 | 59.8% | 256/256 |
| Smoothing | 38/256 | 14.8% | 256/256 |
| Ordinary paraphrasing | 36/256 | 14.1% | 211/256 |
| BIRA | 0/256 | 0.0% | 213/256 |

Human false positives remain 3/256 at the frozen threshold. BIRA has zero
detections among all 213 eligible outputs, but 43 outputs are too short for
the detector and cannot count as demonstrated removal on detectable-length
text. Several rewrites answer embedded quiz instructions instead of preserving
the passage, including outputs such as `yes`, `No` and `None of the above
choices.` Only 170/256 (66.4%) pass the imperfect bidirectional NLI proxy,
versus 222/256 (86.7%) for ordinary paraphrasing. BIRA has no released
repetition-check failures, but two length-cap hits. Its two undefined
perplexities are one-token answers; these are explicitly recorded rather than
silently included in the mean. The results do not establish 100% usable,
meaning-preserving removal. `interpretation.json` records these limitations
and verifies the frozen attack hashes.

## TextSeal evaluation resumption

The stronger-student BIRA result above is not a TextSeal result. In the
faithful TextSeal run `20261103T014241041156Z`, all 256 Llama BIRA rewrites
were saved (two released repetition-check failures, zero length-cap hits),
but the inode-quota failure interrupted owner scoring before any BIRA scores
were saved. The Qwen rewrite arm had not started. Job `1907223` resumes only
cases 7 and 8 from the saved artifacts, without regenerating completed Llama
rewrites. The frozen Python algorithms, inputs and owner threshold are
unchanged; a separately hashed scheduling wrapper moves compiler caches
node-locally. Both cases have now completed successfully, including owner
scoring and all quality rows. Frozen attack hashes were verified.

| TextSeal attack | Detected /256 | Raw evasion | Bidirectional NLI pass |
|---|---:|---:|---:|
| Original marked text | 253 | 1.2% | Not applicable |
| Ordinary Llama paraphrasing | 47 | 81.6% | 65.2% |
| BIRA, released Llama-3.1-8B | 12 | 95.3% | 49.6% |
| Ordinary Qwen3.5-9B paraphrasing | 48 | 81.2% | 80.1% |
| BIRA, Qwen3.5-9B extension | 9 | 96.5% | 65.2% |

The owner threshold is unchanged, with human false positives 1/256 (0.39%).
These raw evasion rates are not meaning-preserving removal rates. Llama BIRA
has 23/256 outputs of at most ten whitespace-separated words; Qwen BIRA has
58/256, versus one such original input. Several outputs answer embedded quiz
questions instead of preserving the passage. Llama has two released
repetition-check failures and Qwen has three, with zero and two length-cap
hits respectively. Undefined perplexities (two Llama, ten Qwen) correspond to
one-token answers and are recorded explicitly. Finite perplexity means are
dominated by outliers; the medians are 12.09 and 16.63. Bidirectional NLI is an
imperfect proxy and can accept short answers; it is not proof of semantic or
factual preservation. `textseal_bira_completed_results.json` records the final
metrics, short-output audit and source artifacts.
