# TextSeal with probability access and blind rewriting

This experiment attacks the original TextSeal sampler on Qwen3.5-4B-Base.
It does not train a student or alter target weights. Existing 256 test responses,
question-disjoint data partitions and the original detector threshold are reused.
No owner detector scores select an attack parameter or candidate.

## Methods and access

* **Direct resampling:** request the top 10 next-token probabilities and sample
  from their renormalized distribution using independent randomness.
* **Smoothing:** implement the unknown-watermark variant of Chang, Hassani and
  Shokri, [Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.264/).
  Estimate confidence as `sum(p_topk**2) + (1-sum(p_topk))**2/(V-k)`, preserving
  the original top-k mass before estimating confidence. Normalize using the
  minimum and maximum on 200 training prefixes, clamp to `[0,1]`, and use the
  resulting confidence raised to alpha as the target mixture weight. The other
  component is a public Qwen3.5-0.8B-Base reference distribution, independently
  truncated to top 10. Fix alpha to 1 and k to 10 before evaluation.
* **BIRA:** implement Hwang, Park and Ok,
  [ICML 2026](https://arxiv.org/abs/2509.23019), Algorithm 1 and Appendix C.
  Use a separate public Qwen3-4B-Instruct-2507 rewriter to compute next-token
  surprisal on an existing response. Suppress the token IDs at or above the
  median surprisal during document rewriting. The rewriter never receives the
  target probabilities, owner key, watermark type or detector scores.
* **Ordinary paraphrasing:** use the same rewriter and prompt with zero bias.

The local probability oracle exposes sampling probabilities after temperature
0.8 and nucleus filtering at 0.95, before TextSeal's keyed selection. It accepts
arbitrary prefixes supplied by the client. This is an explicit API assumption:
it does not hold for an API that returns the conditional keyed selection law or
does not expose probabilities. These probabilities coincide with clean sampling
probabilities for TextSeal, which makes direct resampling an essential baseline.
The attack functions receive only top-k IDs/probabilities and vocabulary size.
The experiment runner owns the model and detector; this is an interface-based
local API simulation, not an operating-system isolation boundary or hosted API
benchmark. The methods do not require knowing the watermark technology.

Both full PDFs were downloaded and read, including appendices: smoothing 27
pages and BIRA 30 pages. Official code is archived under watermarking resources
at revisions `5acda5f1f27ddebe758d051537b0a59982f89b22` and
`6f62ecce6f3410514fd43a40583a4059822af73a`, respectively. The implementation
uses the paper formulas, with a sparse mixture instead of vocabulary-sized
scatter buffers. Smoothing uses continuous confidence normalization from the
paper rather than the official code's 100-bin approximation. Increasing alpha
reduces target weight for confidence in `(0,1)`, as the formula and ablations
state; contrary prose in the paper is not followed.

## Calibration and evaluation

BIRA uses the authors' exact system prompt, extracted without importing their
detector-coupled runner. An explicit user-message wrapper quotes document
content and tells the rewriter to preserve embedded questions and instructions
rather than follow them. An initial sanity batch exposed this instruction/data
confusion; that rewrite lane was canceled before detector evaluation. The
corrected prompt is checked on a synthetic quoted question and a training
document, and its bias calibration is repeated. Completed probability outputs
are reused only after verifying identical probability code, inputs and settings,
with original artifact hashes and signatures recorded in `reuse.json`.
The new public rewriter replaces the paper's Llama
models. Its initial negative bias is calibrated on 50 clean HC3 training
documents, replacing C4. Sweep -1 through -12, selecting the first bias with
any degenerate output, or -12 if none degenerate. Generation uses temperature
0.7, top-p 0.95, top-k 50, and at most 1,500 new tokens. A distinct-unigram ratio
below 0.25 in the final 450 tokens triggers a restart with bias increased by
0.125 toward zero. Remaining degeneration, length caps and restart counts are
reported. This is a method adaptation, not an exact paper benchmark reproduction.

Owner evaluation waits for all frozen attack artifacts. It reports detection
counts, exact binomial intervals, AUROC and human test false positives at the
original validation-calibrated threshold. All fixed attack arms are reported.
The retained 256 test documents contain 255 distinct question IDs. Splits remain
question-disjoint; document-level binomial intervals assume independence and do
not account for the one repeated question group.
Probability attacks produce new continuations of the same prompts, so their
quality is assessed by likelihood, length and repetition rather than similarity
to a particular previous continuation. BIRA and ordinary paraphrasing must
preserve their source response; evaluation additionally measures bidirectional
NLI and numeric-token overlap and exports 20 seeded examples for human review.
The independent DeBERTa NLI model truncates pairs to 512 tokens, and the fraction
affected is reported. These automatic checks do not certify full factual fidelity.

No success rate on TextSeal is inferred from either paper's other watermarks.
BIRA's headline >99% evasion uses default detector thresholds. Its stricter
Table 18 reports 1.2%-8.4% remaining detection at 1% FPR with the 8B rewriter.
Cross-paper percentages are not matched comparisons.

## Commands and execution

From this repository root, with the existing optional watermarking environment:

```bash
PYTHONPATH=. uv --no-cache run --no-sync python scripts/textseal_api_attacks.py prepare
PYTHONPATH=. uv --no-cache run --no-sync python scripts/textseal_api_attacks.py submit RUN/manifest.json
PYTHONPATH=. uv --no-cache run --no-sync python -m pytest ai_content_detector/tests/test_text_attacks.py -q
```

Preparation freezes source, settings, input hashes, public model revisions and
paper provenance. Slurm runs a GPU preflight, then probability attacks, ordinary
paraphrasing and bias calibration in parallel. BIRA follows calibration; owner
evaluation waits for every attack branch. At most three GPUs are used. Each job requests one H100, dev QoS and two
hours. Generation and evaluation save atomic progress after batches/documents.
A warning five minutes before timeout requests a checkpoint and automatic
requeue; a cancellation SIGTERM saves progress without requeueing. Unfinished
batches may be recomputed. Receipts distinguish running, paused, failed and
completed stages. The runner rejects source, configuration and input drift.

Model downloads add Qwen3-4B-Instruct-2507, Qwen3.5-0.8B-Base and the independent
DeBERTa NLI checkpoint to the existing Hugging Face resource cache. No new Python
dependencies are introduced. Pending jobs and CPU unit checks are not empirical
attack results.

The current 2026-11-02 run is
`results/research_workflow/textseal_api_attacks/20261102T153243140377Z`.
Its jobs are GPU preflight `1858903`, probability receipt validation `1858905`,
bias calibration `1858906`, ordinary paraphrasing `1858907`, BIRA `1858908`,
and owner evaluation `1858909`. Both 256-response probability arms are complete
and reused from `20261102T151207546628Z`; their probability implementation and
settings are unchanged. Superseded runs retain their audit records and have no
active downstream jobs. The initial GPU preflight reproduced two original
detector scores, checked target/reference vocabularies and incremental caches,
and ran the rewriter and NLI model. The revised preflight additionally checks
document framing on non-test content.
The focused attack and existing watermarking tests passed together: 15 tests.

## Completed results, checked 2026-11-03

All six Slurm stages completed with exit code zero. Each attack has 256 outputs;
none are empty. The corrected preflight preserved the quoted question instead
of answering it and reproduced the original detector scores.

| Condition | Detected by TextSeal | Detection rate |
| --- | ---: | ---: |
| Original watermarked responses | 253/256 | 98.8% |
| Direct top-10 resampling | 4/256 | 1.6% |
| Smoothing | 9/256 | 3.5% |
| Ordinary paraphrasing | 200/256 | 78.1% |

All rows use the unchanged validation threshold. Human test false positives
are 1/256 (0.39%). Direct resampling has the lowest observed detection;
smoothing provides no observed improvement in this run. Their AUROCs against
human text are 0.503 and 0.513, respectively. These remain results for the
explicit pre-keyed-sampler probability interface, not a hosted API measurement.

The previous Qwen3-4B BIRA adaptation and its metrics were deleted at the
user's request. It has been replaced with the authors' released attack code,
standard Llama-3.1-8B-Instruct replication controls, and an explicitly permitted
Qwen3.5-9B extension. See [the replication protocol](bira_replication.md).

Retained probability/control outputs and quality measurements are in the
run's `results.json`, `*_evaluation.json`, `*_quality.json` and
`semantic_audit.json`. `sanity_check.json` records artifact-integrity checks and
additional failure diagnostics. BIRA generations, BIRA metrics, calibration outputs and mixed-artifact BIRA
fields have been removed; the probability and ordinary-paraphrase controls remain.

The separate faithful rerun `bira_replication/20261103T014241041156Z` completed
both TextSeal BIRA arms in job `1907223`. Remaining detection is 12/256 (4.7%)
with the released Llama-3.1-8B setup and 9/256 (3.5%) with the Qwen3.5-9B
extension, versus 253/256 (98.8%) before attack. Human false positives stay
1/256. Bidirectional NLI pass rates are only 49.6% and 65.2%, and the Qwen arm
has 58 outputs of at most ten words, often short answers to embedded questions.
These are raw detection-evasion results, not 95%-96% usable-removal claims.
See the [completed replication results](bira_replication.md#textseal-evaluation-resumption)
for the new paraphrasing controls, quality limitations and saved artifacts.
