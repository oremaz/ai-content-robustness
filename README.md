# AI Content Detection and Evasion Research

This subproject is a research workbench for three related questions:

1. Can a model distinguish AI-generated content from human-created content?
2. Can a generator learn to evade that detector while preserving the requested
   meaning or image prompt?
3. What happens when the detector is retrained on the attacker's new outputs?

It supports text and image detectors, detector ensembles, text and image evasion
methods, repeated attacker and defender updates, leakage-aware benchmarks, and a
Streamlit scoring application.

AI-content detection is statistical evidence, not proof of authorship. A score
can change with language, topic, generator, decoding settings, image processing,
and adversarial rewriting. Use this project to run controlled experiments or to
assist review, not as the sole basis for a consequential decision.

The bibliography is in [`references.bib`](references.bib). This README explains
the ideas needed to understand the implementation without requiring the reader
to first read those papers.

## The problem in plain language

AI generators leave patterns, but no single pattern is universal. A text model
might produce unusually predictable token sequences. An image generator might
leave frequency artifacts or react differently to denoising. A supervised
classifier can learn such patterns from examples, while a zero-shot detector can
derive a statistic from a pretrained model without training a new classifier.

An adaptive attacker makes the problem harder. It observes a detector score and
changes the generator to lower that score. A defender can then retrain on these
new evasive samples. This repository represents that loop as:

```text
prompt -> generator -> candidate content -> detector -> AI score
             ^                              |
             |________ evasion reward ______|

fixed evaluation set -> measure attacker -> retrain defender -> measure again
```

The key experimental question is therefore not only "does the detector work on
today's samples?" It is also "does it continue to work on samples optimized
against it, including samples from prompts and detectors that training never
saw?"

## Essential vocabulary

| Term | Meaning in this project |
|---|---|
| **Detector** | A component that maps text or an image to an AI-oriented score and a label. |
| **Supervised detector** | A classifier trained on labeled human and AI examples. It needs a fitted checkpoint. |
| **Zero-shot detector** | A method that computes a statistic using pretrained models without fitting a task-specific detector classifier. It can still require large model weights and a threshold. |
| **Raw statistic** | The detector's native measurement, such as a likelihood ratio, curvature, cosine similarity, or anomaly score. |
| **Calibration** | Learning a mapping from scores to probabilities on a separate representative validation set. |
| **Evasion attack** | A rewrite or generator update intended to reduce detection while retaining useful content. |
| **Attacker** | The generator or transformation being optimized. |
| **Defender** | The detector being evaluated or retrained. |
| **Held-out** | Data or a detector excluded from training and used only to test generalization. |
| **Checkpoint** | Saved trained weights or a fitted classical model. Large project checkpoints are not stored in git. |

## Quickstart

Run every command from the repository root. The project uses the root `uv`
environment and expects `PYTHONPATH=.` for module imports.

### 1. Create the environment

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run python -m spacy download en_core_web_sm
```

On the CUDA 12.4 research VM, install the GPU environment instead:

```bash
uv pip install -r requirements_gpu.in
```

The standard installation is enough for the CPU and mock-based test suite. Most
large detectors and all meaningful training runs need downloaded Hugging Face
weights, project checkpoints, and usually a GPU. On macOS, XGBoost also requires
an OpenMP runtime such as `libomp`.

### 2. Check imports and wiring

```bash
PYTHONPATH=. uv run python -m ai_content_detector.rl_evasion.run_experiments \
  --experiment smoke_test
```

This is deliberately lightweight. It checks imports, configuration objects,
reward plumbing, and trainer construction. It does not download or run every
large model.

### 3. Run the focused test suite

```bash
PYTHONPATH=. uv run python -m pytest ai_content_detector/tests -q
```

These tests validate formulas, tensor shapes, label conventions, split
contracts, failure behavior, seeding, and leakage controls using CPU-scale
fixtures and mocks.

### 4. Launch the scoring app

```bash
PYTHONPATH=. uv run streamlit run ai_content_detector/app.py
```

Open the URL printed by Streamlit, select the text or image tab, choose the
available detectors, provide content, and run the analysis. The app reports why
a detector is unavailable. It does not replace a missing checkpoint with a
heuristic result. At least one usable detector is required.

The app treats text shorter than 50 characters as too short for reliable use.
Some individual methods impose stronger requirements. For example,
Disrupt-and-Recover defaults to at least 35 words.

### 5. Run a small benchmark

Once Hugging Face downloads are available, the following command downloads 20
HC3 examples and runs CPU-compatible attack baselines against a RoBERTa
detector. It is a practical pipeline check, not a publishable benchmark:

```bash
PYTHONPATH=. uv run python \
  -m ai_content_detector.rl_evasion.benchmarking.benchmark \
  --dataset hc3 \
  --max-samples 20 \
  --lightweight \
  --detectors roberta_classifier \
  --output-dir results/benchmark_quickstart
```

`--lightweight` changes the attack methods, not the scoring stack. This command
still downloads the RoBERTa detector, the E5 semantic encoder, HC3, and NLTK
resources. Use the smoke test and unit tests for an offline CPU check.

HC3 contains aligned prompts, human answers, and ChatGPT answers. The
CNN/DailyMail option contains prompts and human continuations but no stored AI
continuations, so it can evaluate generation methods but skips rewrite-only
methods that require an existing AI text.

## Repository map

```text
ai_content_detector/
├── app.py                              # Streamlit text and image UI
├── detectors/
│   ├── ensemble.py                    # Detector interface and aggregation
│   ├── text_detectors.py              # Text detectors
│   ├── image_detectors.py             # Image detectors
│   └── style_detector.py              # LUAR few-shot style detector
├── rl_evasion/
│   ├── config.py                      # Reproducible experiment configs
│   ├── run_experiments.py             # Unified experiment CLI
│   ├── text_evasion/                  # GRPO, SPIN, rewards, features, HUMPA
│   ├── image_evasion/                 # DDPO training and evaluation
│   ├── arms_race/                     # Repeated adaptation and MAML
│   └── benchmarking/                  # Datasets, baselines, runner
├── notebooks/explainable_detector.ipynb
├── references.bib
└── tests/
```

Internal supervised detectors reuse checkpoint loading and model wrappers from
`ml_pipeline/`. Shared checkpoint behavior should remain there rather than being
duplicated in this subproject.

## Detector interface and score semantics

Every detector implements `BaseDetector`:

```python
result = detector.detect(content)

result.score          # float in [0, 1], oriented so larger means more AI-like
result.label          # "human", "ai", "uncertain", or "error" in an ensemble
result.detector_name  # runtime detector name
result.details        # raw statistic, thresholds, semantics, or failure context
```

`is_available()` checks dependencies, model configuration, support data, and
checkpoints. `unload()` releases known model references and clears CUDA cache
when possible. Heavy models are loaded lazily.

### A score is not automatically a probability

All public scores use the same direction, with `0` more human-like and `1` more
AI-like. That common range makes display and aggregation convenient, but it does
not give every number the same interpretation.

- `classifier_probability` comes from a fitted classifier's softmax or
  `predict_proba`. It is still only as reliable as that classifier's data.
- `calibrated_probability` means a separate calibrator was fitted on labeled
  validation examples.
- `uncalibrated_*` means the value is a normalized or monotonic display score.
  It preserves ordering but must not be read as confidence.
- A raw statistic and its decision threshold are retained in `details` when a
  paper method makes its decision in the raw space.

Inspect `result.details["score_semantics"]` whenever it is present. For example,
a Binoculars display score of `0.8` does not mean an 80 percent probability of
AI authorship unless a calibrator was explicitly supplied.

The generic three-way helper labels scores at least `0.65` as `ai`, at most
`0.35` as `human`, and the interval between them as `uncertain`. Methods with a
native paper threshold make their label from the raw paper statistic instead.

### Ensemble behavior

`EnsembleAggregator` runs all selected detectors independently. Its default
weighted average is

\[
S_{\mathrm{ensemble}} = \sum_{j=1}^{m} \tilde{w}_j S_j,
\qquad
\tilde{w}_j = \frac{w_j}{\sum_k w_k}.
\]

Without a fitted logistic calibrator, this is an uncalibrated normalized score
and the label comes from detector votes. With a calibrator fitted on disjoint
labeled validation data, the aggregate is reported as a calibrated probability.
An empty ensemble, a calibration set with one class, and a run in which every
detector fails all raise clear errors. A failure from one detector is exposed in
the per-detector results and excluded from the uncalibrated aggregate.

## How the text detectors work

### Background: tokens, probability, surprisal, and perplexity

A language model assigns a conditional probability to each next token. For a
token sequence \(x_1,\ldots,x_T\), the probability of token \(x_t\) depends on
the preceding tokens \(x_{<t}\). Its surprisal is

\[
s_t = -\log p(x_t \mid x_{<t}).
\]

A predictable token has low surprisal. Perplexity is the exponential of mean
surprisal. Several detectors start from the observation that machine-generated
text can be unusually predictable or have a different surprisal pattern from
human text. This is a population tendency, not a rule for every document.

### Supervised project baselines

| Detector | Signal and implementation | Required resource |
|---|---|---|
| `ModernBERTDetector` | A QLoRA-fine-tuned ModernBERT binary classifier. The project checkpoint follows the MAGE label order `0=AI`, `1=human`. | `bench_aitextdetect` ModernBERT checkpoint |
| `TFIDFDetector` | Counts weighted word features with TF-IDF, then applies logistic regression. It is a useful classical baseline for surface vocabulary patterns. | Fitted classifier and `tfidf_vectorizer.pkl` |

QLoRA trains small low-rank adapter parameters while keeping a quantized base
model mostly frozen. This reduces training memory. At inference time it is still
a learned classifier and can inherit the domains and generators represented in
its training data.

TF-IDF gives a word more weight when it is frequent in a document but uncommon
across documents. Logistic regression learns a linear boundary over those
features. It is cheap and interpretable, but paraphrasing and domain shift can
change its evidence substantially.

### Binoculars

`BinocularsDetector` compares two related causal language models: an observer
and a performer. Let \(q_t\) be the observer distribution at position \(t\),
\(p_t\) the performer distribution, and \(x_t\) the observed token. The
implemented ratio is

\[
B(x) =
\frac{\frac{1}{T}\sum_t -\log q_t(x_t)}
     {\frac{1}{T}\sum_t \sum_v -p_t(v)\log q_t(v)}.
\]

The numerator measures how surprised the observer is by the actual text. The
denominator measures how surprised the observer would be by the performer's
full predictive distribution. AI text tends to yield a lower ratio for the
paired models used by the method. The default low-FPR threshold is `0.9015`.
The label is decided from this raw ratio. A sigmoid is used only to display the
score on `[0, 1]` unless a calibrator is provided.

Default models are `tiiuae/falcon-7b` and
`tiiuae/falcon-7b-instruct`. Availability requires CUDA and both model
configurations and weights.

### Fast-DetectGPT

`FastDetectGPTDetector` measures conditional probability curvature without
generating perturbed documents. Let \(\ell_t(v)\) be the scoring model's log
probability for vocabulary token \(v\), and let \(r_t(v)\) be the reference
model's distribution. The implemented statistic standardizes how much more
likely the observed tokens are than tokens expected under the reference model:

\[
C(x) =
\frac{\sum_t \left[\ell_t(x_t) -
      \mathbb{E}_{v\sim r_t}\ell_t(v)\right]}
     {\sqrt{\sum_t
      \operatorname{Var}_{v\sim r_t}[\ell_t(v)]}}.
\]

Higher curvature is treated as more AI-like. The scoring and reference models
may differ, but they must use identical token vocabularies. The raw curvature
and threshold remain in `details`; the `[0, 1]` score is an uncalibrated
logistic transform.

### DivEye

`DivEyeDetector` turns the surprisal sequence into nine document features:
mean, variance, skewness, kurtosis, first-difference mean and variance,
second-difference variance and entropy, and lag-one second-difference
autocorrelation. First differences describe how surprise changes from one token
to the next. Second differences describe how that change itself accelerates or
reverses.

Those nine features are inputs to XGBoost. The fitted classifier must preserve
the feature schema and the label convention `0=AI`, `1=human`. The detector is
not usable with only the language model because the decision boundary is learned
by the missing classifier artifact. Text is truncated to at most 1024 tokens.

### Disrupt-and-Recover

`DisruptRecoverDetector` tests whether a passage can be reconstructed after a
controlled corruption:

1. Split the text at punctuation boundaries.
2. Shuffle all word tokens inside each chunk with a fixed seed.
3. Ask one recovery model call to restore the original order without adding or
   removing words.
4. Compare the recovery with the source using BERTScore for semantic similarity
   and averaged Kendall and Spearman rank correlation for structural order.
5. Average the semantic and structural signals, then map that similarity to an
   AI-oriented score.

The intuition is that more concentrated or regular machine text may be easier
for a model to reconstruct exactly. The implementation needs either an injected
`recover_fn` or an OpenAI-compatible model configured with
`DR_RECOVERY_MODEL` and an API key. Real BERTScore inference also downloads a
transformer checkpoint.

### Window smoothing

`WindowSmoothedTextDetector` splits a long document into overlapping word
windows, scores each with another detector, iteratively pulls neighboring
scores toward one another, downweights unstable early windows, and computes a
weighted document score. It is useful when AI-like evidence is local rather
than uniform across a document.

This is a practical document-window variant. It is not the cited paper's
learned token-level Markov random field. `MarkovCalibratedTextDetector` remains
only as a compatibility alias.

### IPAD

`IPADDetector` asks whether a text is consistent with an inferred generating
prompt. It uses three released LoRA adapters on
`microsoft/Phi-3-medium-128k-instruct`:

1. Prompt inversion predicts a likely source prompt from the candidate text.
2. PTCV scores whether that inferred prompt is consistent with the text.
3. RC scores the relation between the input and a separately regenerated text.
4. Full `Yes` and `No` sequence likelihoods are normalized as a binary
   distribution.
5. The two signals are fused with PTCV weight `0.45`; the paper threshold is
   `0.54`.

Regeneration is a separate operation, not an adapter role. The shared base model
is about 14 billion parameters, so the detector refuses CPU loading and expects
a sufficiently large GPU plus the base model and all three adapters.

### Paraphrase round trip and LUAR style

`ParaphraseRoundTripDetector` rewrites a passage and measures semantic distance
between the original and paraphrase. It is an explicit heuristic baseline,
inspired by the idea that generated text may react differently to another
rewrite. `InversionDetector` is its compatibility alias.

`StyleEmbeddingDetector` uses LUAR to map documents into a vector space where
cosine similarity represents writing style. Before detection, callers must
provide both human and AI support examples with `setup_support_set()`. Detection
compares a query with the two support-set centroids. This is few-shot detection,
not a universal pretrained AI centroid.

## How the image detectors work

| Detector | Main idea | Repository relationship |
|---|---|---|
| `EfficientNetDetector` | Convolutional classifier trained to separate fake and real images. | Internal `bench_imai_artifact` checkpoint |
| `CLIPImageDetector` | Binary classifier built on CLIP image representations. | Internal `bench_imai_artifact` checkpoint |
| `DINOv2Detector` | Binary classifier built on self-supervised DINOv2 features. | Internal `bench_imai_artifact` checkpoint |
| `SigLIPDetector` | Hugging Face image-classification pipeline whose class names define fake and real semantics. | Configured model-card behavior |
| `PatchBasedClassifier` | Scores overlapping local crops with another detector and aggregates them. | Project robustness wrapper |
| `FrequencyDetector` | Measures a single-level Haar high-frequency to low-frequency energy ratio. | Explicit cheap heuristic |
| `MLEPDetector` | Builds multi-scale local entropy maps after spatial patch permutation, then classifies them. | Paper method, trained CNN required |
| `WaRPADDetector` | Measures DINOv2 representation sensitivity after removing high-frequency content. | Paper method, raw score unless calibrated |
| `DenoisingTrajectoryDetector` | Measures feature consistency along diffusion inversion and recovery states. | DTAD method implementation |

### Frequency information and Haar wavelets

An image can be decomposed into a low-frequency component, containing broad
color and shape, and high-frequency components, containing edges and fine
texture. A Haar wavelet transform performs this split with local sums and
differences.

`FrequencyDetector` computes a one-level high-to-low energy ratio and squashes
it into `[0, 1]`. It is fast, but its score can also react to compression,
sharpening, resizing, or natural texture. It is intentionally a heuristic and
is not WaRPAD.

### MLEP

`MLEPDetector` searches for local entropy patterns:

1. Independently permute non-overlapping `2 x 2` spatial patches in each color
   channel.
2. Repeat at image scales `1`, `0.5`, and `0.25`.
3. Compute exact stride-one `2 x 2` local entropy maps.
4. Stack the maps and classify them with a trained ResNet-50.

Entropy measures local uncertainty in the shuffled pattern. The preprocessing
alone does not produce a valid AI decision, so a fitted CNN checkpoint is
mandatory.

### WaRPAD

`WaRPADDetector` resizes an image to `896 x 896`, divides it into sixteen
non-overlapping `224 x 224` patches, removes high-frequency content with a
two-level Haar transform using `alpha=0.1`, and compares original and perturbed
DINOv2 ViT-L/14 CLS embeddings. Large representation changes indicate an
anomalous sensitivity to frequency removal.

Without a supplied threshold or calibrator, the class returns an unclassified
raw anomaly score and the label `uncertain`. This avoids inventing a universal
decision boundary.

### Denoising trajectory detection

`DenoisingTrajectoryDetector` maps an image into a diffusion model's latent
space, performs deterministic DDIM inversion toward noisier states, reconstructs
predicted-clean images along the trajectory, and extracts CLIP ViT-L/14
layer-15 image features. It aggregates similarity across the trajectory. The
working hypothesis is that real and generated images follow different recovery
geometry.

The output is an uncalibrated similarity transform. Exact numerical parity with
author code and weights has not been established.

## Evasion methods

### The reward design

Text and image attacks optimize more than detector evasion. If the only reward
were `1 - detector_score`, a generator could emit empty or irrelevant content.
The text configuration therefore combines detector evasion, semantic
similarity, and a length and quality term:

\[
R_{\mathrm{text}} = 0.6R_{\mathrm{evasion}}
                  + 0.3R_{\mathrm{semantic}}
                  + 0.1R_{\mathrm{quality}}.
\]

The image default combines detector evasion and CLIP prompt alignment with
weights `0.7` and `0.3`. Its aesthetic weight is zero because no trained
aesthetic model is bundled. Missing or invalid detector rewards fail clearly.

### GRPO text evasion

`GRPOTextEvasionTrainer` samples a group of completions for each prompt, scores
them, and uses reward relative to the group as the learning signal. In simple
terms, completions that evade the detector better than their siblings become
more likely, and worse completions become less likely. A KL penalty or reference
policy limits movement away from the starting model.

The configured generator is adapted with LoRA. The method is called GRPO only
when the TRL implementation is available. The optional local fallback is
explicitly named group-normalized REINFORCE and is disabled by default.

### SPIN with feature monitoring

`SPINTrainerWithFeatureMonitoring` alternates between generating model outputs
and preferring human reference continuations over those outputs. Its loss uses
completion-only conditional log-probabilities, so prompt tokens do not dominate
the preference signal. The frozen reference snapshot advances after each SPIN
iteration.

Stylometric statistics, E5 embeddings, LUAR style embeddings, and learned probe
features are monitoring outputs. They are not gradient losses because decoding
text breaks a differentiable path from those features back to model parameters.
`MultiSPINTrainer` remains as a compatibility alias.

### HUMPA-style proxy intervention

`ProxyEvasionWrapper` changes next-token logits during decoding:

\[
z_{\mathrm{combined}} = z_{\mathrm{target}}
  + \alpha\left(z_{\mathrm{fine\ proxy}} - z_{\mathrm{reference\ proxy}}\right).
\]

The difference isolates what the fine proxy learned relative to its frozen
reference and steers the target without fine-tuning the target itself. All three
models must share the same vocabulary. This wrapper implements the decoding-time
intervention; training the fine proxy is outside its scope.

### Adversarial paraphrasing

`AdversarialParaphrasingBaseline` performs autoregressive candidate search. At
each token position it keeps candidates from top-p `0.99`, limits them to top-k
`50`, decodes each candidate prefix, asks a supplied detector for an AI score,
and selects the lowest-scoring token. It requires a real guidance detector.
This is much more expensive than ordinary sampling because every output token
can trigger many detector calls.

The benchmark also includes simpler controls: no transformation, neural
paraphrasing, WordNet synonym substitution, human-style prompt instructions,
high-temperature sampling, and optional OpenRouter generation.

### DDPO image evasion

`DDPOImageEvasionTrainer` treats diffusion denoising decisions as a policy. It
samples complete denoising trajectories, scores the final images, and increases
the likelihood of transitions from trajectories with above-baseline rewards.
The implementation uses stochastic DDIM transition likelihoods, scheduler
prediction-type handling, per-prompt advantage statistics, PPO-style ratio
clipping, seeded sampling, and gradient accumulation.

### MAML fast adaptation

`MAMLAdaptation` learns a LoRA initialization intended to adapt quickly to a new
detector. Each meta-training episode uses disjoint support and query prompts:

1. Take several policy-gradient steps against sampled detectors on support
   prompts.
2. Evaluate the adapted fast weights on different query prompts.
3. Backpropagate the query loss through the inner updates to improve the shared
   initialization.

Full MAML retains second-order derivatives through the inner loop. `--first-order`
uses FOMAML, which drops those Hessian terms to reduce compute and memory. One
detector is reserved before meta-training, and held-out adaptation is compared
with the exact pre-meta initialization.

### Repeated attacker and defender adaptation

`ArmsRaceExperiment` keeps a fixed evaluation prompt split and records paired
measurements at three points in each round: before attacker training, after the
attacker update, and after defender retraining. The attacker step budget is
enforced.

`AdaptiveClassifierDefender` retrains a RoBERTa classifier on current attacker
outputs and matched human references. `RADARDefender` is only a compatibility
alias; this is not the RADAR alternating paraphraser and detector algorithm.
The resulting curves are empirical repeated-update measurements, not a Nash
equilibrium or Nash gap.

Image arms-race mode is intentionally rejected because an adaptive image
defender is not implemented.

## Evaluation and leakage controls

### Data flow

Text and image prompt sets are split deterministically before training. The
evaluation portion is immutable. MAML additionally separates support prompts,
used for inner adaptation, from query prompts, used for its meta-objective.

Benchmark datasets validate one-to-one alignment between prompts, AI texts,
human references, and sample identifiers. They record a source revision when
the dataset exposes one. Trained GRPO and SPIN checkpoints record method,
optimized detector names, and seed metadata. The benchmark rejects an attack
checkpoint whose optimized detector names overlap declared held-out detectors.
Training sample IDs are not yet stored in attack checkpoints, so sample-level
overlap cannot yet be rejected automatically.

Methods declare whether they rewrite an existing text, generate a new text from
a prompt, or support both. Incompatible dataset and method pairs are reported as
skipped. Failed transformations are counted and excluded; the original text is
never substituted as a successful evasion output.

### Metrics

- **AUROC** measures how often a random AI example receives a higher score than
  a random human example across all thresholds. It requires both classes.
- **False-positive rate (FPR)** is the fraction of human examples incorrectly
  labeled AI at a threshold.
- **True-positive rate (TPR)** is the fraction of AI examples correctly labeled
  AI at that threshold.
- **TPR at a target FPR** uses the best empirically achievable threshold at or
  below the requested FPR. It does not interpolate an unattainable operating
  point.
- **Attack success rate** is the fraction of generated examples whose mean
  evasion reward exceeds `0.5`, equivalently whose mean detector AI score is
  below `0.5`. It does not require that the original example was first detected.
- **Semantic similarity** or **CLIP alignment** checks that successful evasion
  did not simply discard the requested content.

Empty, misaligned, non-finite, and invalid single-class metric inputs are
rejected rather than turned into plausible-looking results.

### Feature and notebook leakage controls

`FeatureBank` selects probes on a group-disjoint validation partition and fits
its scaler on training data only. The explainability notebook assigns stable
document identifiers before paragraph expansion, uses document-disjoint train,
validation, and test sets, restores sentence segmentation, labels SHAP direction
for the actual class, and reserves test data for final evaluation.

## Running experiments

The unified entry point is:

```bash
PYTHONPATH=. uv run python -m ai_content_detector.rl_evasion.run_experiments \
  --experiment <name>
```

| Experiment | Example | Main requirement and output |
|---|---|---|
| Smoke test | `--experiment smoke_test` | CPU wiring check; no training |
| GRPO text | `--experiment grpo_text --model Qwen/Qwen3.5-9B-Base --epochs 3` | Local causal LM, TRL, detectors, GPU; saves a final adapter and metadata |
| SPIN | `--experiment multispin --epochs 5` | Local causal LM, CNN/DailyMail, GPU; saves one checkpoint per iteration |
| DDPO image | `--experiment ddpo_image --epochs 50` | Stable Diffusion, image detectors, GPU; saves diffusion LoRA checkpoints |
| Arms race | `--experiment arms_race --modality text --rounds 10` | Working text attacker and adaptive defender; saves round history |
| Meta-adaptation | `--experiment meta_adapt --epochs 100` | At least two functioning detectors, LoRA model, GPU; saves `meta_init` |
| Benchmark | `--experiment benchmark --dataset hc3 --epochs 200` | Here `--epochs` is used as the maximum sample count; saves benchmark JSON |

Shared options are `--model`, `--epochs`, `--rounds`, `--modality`,
`--first-order`, `--dataset`, and `--output-dir`. Configuration dataclasses in
`rl_evasion/config.py` expose the remaining learning rates, batch sizes, reward
weights, prompt splits, seeds, and checkpoint intervals.

For finer benchmark control, including explicit held-out detectors and
checkpoints, call the benchmark module directly:

```bash
PYTHONPATH=. uv run python \
  -m ai_content_detector.rl_evasion.benchmarking.benchmark \
  --dataset hc3 \
  --max-samples 200 \
  --detectors binoculars fast_detect_gpt \
  --heldout-detectors diveye \
  --grpo-checkpoint results/text_evasion/grpo/final \
  --output-dir results/benchmark
```

OpenRouter baselines are added only when `OPENROUTER_API_KEY` is present. API
failures are counted as failures and never fall back to returning the prompt.

## Required models and external resources

| Component | What must be supplied |
|---|---|
| ModernBERT and TF-IDF | Artifacts under the existing `ml_pipeline` `bench_aitextdetect` checkpoint registry |
| EfficientNet, CLIP, and DINOv2 classifiers | Artifacts under `bench_imai_artifact` |
| Binoculars and Fast-DetectGPT | Their Hugging Face causal language models; CUDA is required by availability checks |
| DivEye | Scoring LM plus fitted XGBoost classifier with the nine-feature schema |
| Disrupt-and-Recover | Recovery callable or OpenAI-compatible API, plus a BERTScore transformer checkpoint |
| IPAD | Phi-3 base model, three released adapters, and a large GPU |
| LUAR style detector | LUAR weights plus task-specific human and AI support sets |
| MLEP | Trained ResNet-50 classifier checkpoint |
| WaRPAD | DINOv2 ViT-L/14 weights, plus a validation threshold or calibrator for classification |
| Denoising trajectory | Stable Diffusion and CLIP ViT-L/14 weights |
| GRPO, SPIN, MAML, and DDPO | Local trainable base models, datasets, functioning reward detectors, and GPU memory |

Useful optional environment variables are:

```bash
export OPENROUTER_API_KEY=...   # OpenRouter benchmark or D&R recovery
export OPENAI_API_KEY=...       # OpenAI-compatible D&R recovery
export OPENAI_BASE_URL=...      # Alternate OpenAI-compatible endpoint
export DR_RECOVERY_MODEL=...    # Recovery model ID for D&R
```

Do not commit keys. Model and dataset downloads also require network access and,
for gated models, the provider's authentication and license acceptance.

## Extending the project

### Add a detector

1. Subclass `BaseDetector` in the appropriate detector module.
2. Set `name` and `modality`.
3. Implement `detect()` and return a `DetectionResult` with an AI-oriented score.
4. Put the native statistic, threshold, class order, and `score_semantics` in
   `details`.
5. Implement a truthful `is_available()` and rely on `unload()` or extend its
   resource list for heavy state.
6. Add focused tests for boundary decisions, batches, missing resources, label
   ordering, and score direction.
7. Register it in the app or experiment detector zoo only after its dependencies
   and failure behavior are explicit.

### Add an evasion method

Declare whether it rewrites existing content or generates from prompts. Preserve
sample identifiers in training metadata, record every detector optimized by the
method, and separate training data from final evaluation. A failed example must
remain a failed example rather than being replaced by the unmodified input.

## Validation status

The current checkout's local validation pass completed the full CPU and
mock-based suite:

```text
179 passed in 5.83s
```

Focused WaRPAD, MAML, and benchmark-contract tests passed `15` tests. Focused
Disrupt-and-Recover tests passed `5` tests after installing the declared
`bert-score==0.3.13` dependency. The following checks also passed:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python \
  -m ai_content_detector.rl_evasion.run_experiments --experiment smoke_test

PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python \
  -c 'import ai_content_detector.app'

PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall -q \
  ai_content_detector

PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python -m json.tool \
  ai_content_detector/notebooks/explainable_detector.ipynb

git diff --check
```

These results establish formula, interface, boundary, and leakage behavior for
the tested paths. They do not establish parity with published metrics.

Local end-to-end validation was limited by unavailable Hugging Face access,
absent internal checkpoints, and a macOS XGBoost load failure caused by missing
`libomp`. The Streamlit module imported successfully and reported those missing
resources, but a real browser session with loaded checkpoints was not run. The
explainability notebook is valid JSON and was inspected, but its dataset loading,
XGBoost training, SHAP plots, and metrics were not executed in this checkout.

## Next steps before relying on research results

The following work is intentionally still open. It replaces the former separate
limitations document and is ordered from validation fundamentals to new
capabilities.

1. **Run real-model validation on the research VM.** Exercise Binoculars,
   Fast-DetectGPT, DivEye, IPAD, LUAR, WaRPAD, and denoising-trajectory inference
   end to end. Run GRPO, SPIN, MAML, DDPO, and repeated adaptation on a GPU. Run
   the Streamlit app in a browser with real checkpoints.

2. **Build immutable benchmark manifests and reproduce paper protocols.** Pin
   dataset revisions, model revisions, preprocessing, prompts, seeds, and
   calibration artifacts. Then reproduce AUROC, TPR, FPR, attack-success, and
   convergence measurements. No published metric has yet been reproduced in
   this checkout.

3. **Train and version the missing artifacts.** Produce the DivEye XGBoost
   classifier, MLEP ResNet-50, internal text and image checkpoints, LUAR support
   sets, and per-detector and ensemble calibrators. Store their schemas, class
   order, training sample identifiers, and provenance beside each artifact.
   Extend attack checkpoint metadata and benchmark checks so held-out sample IDs
   are rejected automatically, just as held-out detector overlap is today.

4. **Calibrate for each deployment domain.** Fit thresholds and calibrators on
   representative validation data disjoint from training and final testing.
   Recheck them across language, topic, generator family, decoding strategy,
   image post-processing, and attack type. A universal calibration artifact
   would be misleading.

5. **Validate external-resource methods.** Run real BERTScore plus network
   recovery for Disrupt-and-Recover. Establish DTAD parity against a complete
   runnable primary reference if one becomes available. Until then, retain its
   raw similarity semantics and avoid paper-parity claims.

6. **Execute the explainability notebook on the VM.** Materialize the
   document-disjoint splits, train XGBoost, generate SHAP analyses, and save the
   exact dataset and model provenance before relying on any plots or metrics.

7. **Complete deliberately out-of-scope method pieces if needed.** Train the
   fine proxy used by HUMPA, add a validated aesthetic reward before assigning
   it nonzero weight, and implement an adaptive image defender before enabling
   image arms-race experiments.

8. **Keep practical variants named honestly.** Window smoothing is not a
   token-level MRF, adaptive RoBERTa retraining is not RADAR, SPIN feature signals
   are monitoring-only, and paraphrase round-trip and wavelet energy are
   heuristic baselines. Compatibility aliases may remain for callers, but new
   results should use the precise runtime names.

9. **Test reproducibility across the production stack.** The audited paths seed
   Python, NumPy, Torch, and explicit sampling generators, but bitwise equality
   is not guaranteed across CUDA devices, drivers, mixed-precision kernels,
   distributed training, or third-party model releases. Treat saved provenance
   and statistically consistent reruns as the reproducibility contract.

Until those steps are complete, interpret uncalibrated scores as comparative
signals and the current tests as correctness checks, not proof of real-world or
paper-level performance.
