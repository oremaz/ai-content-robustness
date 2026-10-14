# Text and image detector catalogue

This reference explains detector mechanisms, score interpretation, and
implementation requirements. For the research workflow, experiments, and setup,
see the [main README](README.md).

- [Text detectors](#how-the-text-detectors-work)
- [Image detectors](#how-the-image-detectors-work)

## How the text detectors work

### The example used throughout

The text examples use the following passage. All example scores, token
statistics, and model outputs in this catalogue are hypothetical illustrations,
not measurements from a recorded run. Model settings and formulas describe the
implementation.

Example passage:

> Wild yeast and bacteria eat the sugars in the flour and give off carbon
> dioxide, and the gluten network traps that gas. In the oven the starch sets,
> so the loaf keeps the shape the gas gave it.

Where a detector looks at individual tokens, the first four positions are used:
`" Wild"`, `" yeast"`, `" and"`, `" bacteria"`.

### Background: surprisal and perplexity

A language model predicts the next token from the preceding context. The
**surprise** (or surprisal) of an observed token $x_t$ is
$s_t=-\log p(x_t\mid x_{<t})$: a predictable token has low surprise. For this
passage the opening surprises are

| token | surprise |
| --- | --- |
| `" Wild"` | 0.9 |
| `" yeast"` | 1.8 |
| `" and"` | 0.5 |
| `" bacteria"` | 1.6 |

Perplexity is the exponential of the average surprise, so the average of these
four is about `1.2` nats, or perplexity `3.3`. AI-generated text can have
distinctive predictability or variation patterns, but these are statistical
tendencies, not reliable signatures of authorship in every document.

### ModernBERT (supervised baseline)

`ModernBERTDetector` learns which patterns distinguish human and AI text from
labeled examples. A QLoRA-fine-tuned ModernBERT reads the passage and predicts
its class; QLoRA trains compact adapters on a mostly frozen, quantized model.

Suppose the classifier returns `details = {"prob_ai": 0.83, "prob_human": 0.17}`,
and the score is `probs[0]`, taken directly with no transform.

**Interpretation:** the checkpoint's class order is `0 = machine`,
`1 = human`, so `0.83` is the AI-oriented probability. A high score means the
passage resembles AI examples in the training distribution, **not** that its
authorship is known with that probability. Nothing validates that class order, so
a checkpoint trained the other way round would be silently inverted. The input
is truncated to the model's `max_seq_length`, so later passages never reach the
classifier. Requires the `bench_aitextdetect` ModernBERT checkpoint.

### TF-IDF + logistic regression (supervised baseline)

`TFIDFDetector` detects differences in word usage rather than using a language
model's token probabilities. TF-IDF represents each document by words weighted
more heavily when they are frequent in that document but uncommon across the
corpus. Logistic regression learns which weighted words favor either class.

The weights that pushed this passage toward the AI side would be the sort of
tokens that separate the training classes, and the classifier returns
`probs[0] = 0.62`, again with no transform.

**Interpretation:** `0.62` sits in the `uncertain` band. The signal is cheap and
interpretable, but it can change under paraphrasing or topic shift, and
classifier probabilities are not automatically well-calibrated. If
`tfidf_vectorizer.pkl` is missing the detector reports itself unavailable rather
than fitting a fallback. Requires a fitted vectorizer and classifier.

### Binoculars

`BinocularsDetector` detects AI-generated text by comparing how predictable
the observed tokens are to what a language model would typically generate
in the same context.

It uses two closely related language models:

- **Observer** ($q_t$): assigns probabilities to the actual tokens.
- **Performer** ($p_t$): provides a reference distribution over possible tokens.

At each position, the numerator is the observer's surprise on the actual token,
and the denominator is the cross-entropy between the two predictive
distributions, i.e. the observer's surprise averaged over the performer's
probabilities:

$$
B(x) =
\frac{\frac{1}{T}\sum_t -\log q_t(x_t)}
{\frac{1}{T}\sum_t \sum_v -p_t(v)\log q_t(v)}.
$$

| position | actual surprise $-\log q_t(x_t)$ | expected $\sum_v -p_t(v)\log q_t(v)$ |
| --- | --- | --- |
| `" Wild"` | 0.9 | 1.4 |
| `" yeast"` | 1.8 | 2.2 |
| `" and"` | 0.5 | 1.1 |
| `" bacteria"` | 1.6 | 2.0 |
| mean | 1.20 | 1.68 |

So $B(x) = 1.20 / 1.68 \approx 0.72$.

**Interpretation:** **lower is more AI.** The observed tokens were more
predictable than the performer's reference distribution expected, which is the
hypothesis the method encodes. The default threshold is `0.9015`:
$B(x) < 0.9015$ is labeled **AI-generated**, $B(x) \geq 0.9015$ is labeled
**human-written**, and the label is taken from the raw ratio, so it is always
binary. This is a low-false-positive-rate operating point, not a guarantee. The
display score is `sigmoid((0.9015 - raw) / 0.05)`, which maps `0.72` to about
`0.97` and the threshold itself to `0.5`. A calibrator can replace that map; the
raw ratio and the threshold used are kept in `details`.

Both models are truncated to `512` tokens, and the observer and performer are
not interchangeable: swapping them changes the statistic. The defaults are
`tiiuae/falcon-7b` (observer) and `tiiuae/falcon-7b-instruct` (performer), both
requiring CUDA.

### Fast-DetectGPT

`FastDetectGPTDetector` asks whether the observed tokens are unusually likely
under a scoring language model, relative to plausible alternative tokens. It
compares each actual token's log-probability with the expected log-probability
of tokens drawn from a reference model, then normalizes the difference by its
expected variation. Unlike perturbation-based methods, it does **not** need to
generate rewritten versions of the text.

Let $\ell_t(v)$ be the scoring model's log-probability and $r_t(v)$ the
reference distribution at position $t$:

$$
C(x)=\frac{\sum_t[\ell_t(x_t)-\mathbb E_{v\sim r_t}\ell_t(v)]}
{\sqrt{\sum_t\operatorname{Var}_{v\sim r_t}[\ell_t(v)]}}.
$$

| position | $\ell_t(x_t)$ | $\mathbb E_{v\sim r_t}\ell_t(v)$ | $\operatorname{Var}_{v\sim r_t}\ell_t$ | difference |
| --- | --- | --- | --- | --- |
| `" Wild"` | -0.9 | -1.4 | 0.09 | +0.5 |
| `" yeast"` | -1.8 | -2.2 | 0.09 | +0.4 |
| `" and"` | -0.5 | -1.1 | 0.16 | +0.6 |
| `" bacteria"` | -1.6 | -2.0 | 0.16 | +0.4 |

So $C(x) = 1.9 / \sqrt{0.50} \approx 2.69$.

**Interpretation:** a larger $C(x)$ means the actual tokens are more favored
than typical reference alternatives, relative to the variance; this is treated
as **more AI-like**. Classification compares the raw statistic to its configured
threshold, default `0.0`, so the label is binary and both values are kept in
`details`. The `[0, 1]` display score is a plain `sigmoid(C(x))`, here about
`0.94`, and `score_semantics` is `uncalibrated_logistic_transform`.

Two implementation details matter when reading raw values. The numerator and
the variance are both **sums**, not means, so the statistic grows roughly like
$\sqrt{N}$ with length and is not comparable across passage sizes. And by
default the reference model **is** the scoring model, which reduces the
expectation to the negated entropy of that model; pass an explicit second model
to get the paper's construction. When you do, the two tokenizers must have
identical vocabularies. Inputs are truncated to `512` tokens, and passages
shorter than ten tokens return a score of `0.0`.

### DivEye

`DivEyeDetector` looks at **how token surprise fluctuates across a document**,
not just its average. It computes per-token surprisal and reduces it to nine
features, in this fixed order: mean, variance, skewness, excess kurtosis, mean
and variance of the first differences, variance and entropy of the second
differences (20 histogram bins), and the lag-one autocorrelation of the second
differences. First differences measure changes in surprise; second differences
measure how those changes evolve. XGBoost learns which patterns separate human
from AI text.

The detector returns the classifier's probability for label `0`,
which is resolved by **label lookup** in `classes_` and is not fooled by column
ordering, but the checkpoint must still use `0 = AI`, `1 = human`. A higher
score means the document's surprisal dynamics resemble those of AI training
examples; it is a **learned** judgment, not a universal rule about smoothness.
Passages under five tokens raise; inputs are truncated to `1024` tokens; and the
language model alone is insufficient, the fitted XGBoost artifact is mandatory.
The full feature dict is returned in `details["surprisal_features"]`, so the
classification can always be re-checked against the schema.

### Disrupt-and-Recover

`DisruptRecoverDetector` tests how easily a model can reconstruct the original
word order after scrambling it. The hypothesis is that machine-written prose
may have more regular structure, making its original arrangement easier to
recover.

It splits the text at punctuation boundaries and shuffles the words within each
chunk of at least three words, using a fixed seed so the corruption is
reproducible. For this passage the second chunk becomes:

> that gas the gluten network traps.

The recovery model is asked to restore the order without adding or removing
words, giving back `the gluten network traps that gas.` The result is compared
with the original in two ways: **BERTScore** F1 (`0.94`) for semantic similarity,
and `coverage x mean(Kendall, Spearman)` remapped to `[0, 1]` (`0.83`) for word
order agreement. The two are averaged to give a `recovery_similarity` of `0.885`.

**Interpretation:** greater recovery similarity maps to a more AI-oriented
score, here `sigmoid((0.885 - 0.72) / 0.08) = 0.89`. The `0.72` is the **center
of that logistic**, not a decision boundary: the `0.65` label band is only
reached at a similarity of about `0.77`. This is a heuristic hypothesis, not a
guaranteed property of AI text, and no probability interpretation follows from
the score. Passages under 35 words return `uncertain` with a reason. Requires
an injected `recover_fn` or an OpenAI-compatible model configured with
`DR_RECOVERY_MODEL` and an API key, plus BERTScore weights. The corrupted and
recovered texts are stored in `details`, which makes results large.

### Window smoothing

A **word window** is a consecutive chunk of text. With `window_size = 64` and
`stride = 32`, the detector reads words 1-64, then 33-96, then 65-128, and so on.
The starting position moves forward by 32 words each time, so adjacent windows
share 32 words. Here words are separated by whitespace, not tokenizer tokens.

`WindowSmoothedTextDetector` runs an existing detector on each chunk, adjusts
its score using neighboring chunks' scores, then combines them into one document
score. The purpose is to examine passages individually and reduce isolated score
fluctuations: if one chunk scores strongly AI-like while its overlapping neighbors
score human-like, their scores influence its adjustment. Texts of 64 words or
fewer are scored as a single window.

This is a heuristic wrapper around a detector, not new evidence of authorship.
It can also blur a real change between human and AI passages, and improved
accuracy must be validated. The first two windows receive reduced weights inside
the adjustment and final average; an optional complexity penalty reduces the
score for very simple text. Overlapping words contribute more than once, so this
is not a per-word average. Raw and adjusted scores are returned in `details`.
This implementation is not the cited paper's learned token-level Markov random
field; `MarkovCalibratedTextDetector` is a compatibility alias.

### IPAD

`IPADDetector` asks whether a passage looks like the answer to a plausible
AI prompt. Its key idea is to **infer a prompt from the text**, then check how
well the prompt and the text fit together, including via an independently
regenerated answer.

Suppose the inverter adapter returns the prompt
`Explain why sourdough bread rises.`, and the other two adapters judge that
prompt against the text:

| adapter | question | answer probability |
| --- | --- | --- |
| PTCV | could an AI have generated this text from that prompt? | `p_yes = 0.80` |
| RC | is this text generated similarly to a fresh answer to that prompt? | `p_yes = 0.60` |

Both are two-way renormalized sequence likelihoods over the tokens `" yes"` and
`" no"`, so they sum to `1` by construction. The fused score is
`0.45 * 0.80 + 0.55 * 0.60 = 0.69`.

**Interpretation:** the paper's decision threshold is `0.54` and the comparison
is strict, so `0.69` is labeled **AI**. A higher fused score indicates greater
consistency with the inferred-prompt generation process. This is a
method-specific threshold, not universal proof or a guaranteed calibrated
probability. The fusion gives PTCV weight `0.45` and RC weight `0.55`. Requires the approximately
14B-parameter base model, all three adapters, and a sufficiently large GPU; CPU
loading is refused. Regeneration is a separate operation, not a fourth adapter,
and `details["regeneration_backend"]` records which model produced it.

### Paraphrase round trip

`ParaphraseRoundTripDetector` checks **how far the passage is from plain,
predictable prose**. It asks a generator to rewrite the text "into clear,
standard, and predictable prose", then measures the **word-level** edit distance
between the original and the rewrite, normalized by the longer of the two.

If the rewrite changed 12 of the passage's 34 words, the normalized distance is
`0.35`, and the score is `min(0.35 * 1.5, 1.0) = 0.53`.

**Interpretation:** the mapping is a linear ramp with gain `1.5` that saturates
at a normalized distance of `0.67`, so it is a heuristic normalized distance,
not a semantic similarity and **not** a calibrated probability; an optional
complexity penalty multiplies the score by `0.8` for very simple text. The
direction is the stated one: a large distance means the original was far from
plain prose, which is scored as AI. This is **not** a validated universal
authorship test, and the 20-character minimum is a character count rather than
the 35-word minimum used by Disrupt-and-Recover. The full rewrite is stored in
`details["rewritten"]`. `InversionDetector` is a compatibility alias.

### LUAR style embedding

`StyleEmbeddingDetector` asks whether a document's writing **style** more
closely resembles known human or AI examples. LUAR maps texts into an embedding
space; the method computes human and AI support-set centroids and compares the
query embedding with each using cosine similarity.

For this passage, with a support set of 200 AI and 200 human texts:

| quantity | value |
| --- | --- |
| cosine to AI centroid | 0.71 |
| cosine to human centroid | 0.33 |

The score is a softmax over those two cosines with temperature `0.5`, giving
`exp(0.71/0.5) / (exp(0.71/0.5) + exp(0.33/0.5)) = 0.68`, not the raw cosine.

**Interpretation:** a higher AI-oriented score indicates greater stylistic
similarity to the supplied **AI support examples**, relative to the human ones.
It depends critically on the support set, not on a universal pretrained "AI
style" prototype, and the temperature decides how sharp the boundary is. Both
support classes must be provided with `setup_support_set()` before detection;
the detector will not fabricate a default.

## How the image detectors work

### EfficientNet (supervised baseline)

`EfficientNetDetector` uses convolutional features to classify real and generated
images. The classifier is trained on labeled examples; its spatial filters can
capture textures and artifacts associated with generators in the training data.
This wrapper loads the trained checkpoint rather than training it during scoring.

PIL inputs are resized to `200 x 200` and ImageNet-normalized. The score is the
classifier's probability for class `0 = fake`: higher means more resemblance to
generated examples learned during training, not proof of origin. The wrapper
assumes that class order without validating it. Requires the internal
`bench_imai_artifact` EfficientNet checkpoint.

### CLIP image classifier (supervised baseline)

`CLIPImageDetector` uses pretrained CLIP image representations with a binary
classifier trained on labeled real and generated images. The fitted classifier
learns which features distinguish the classes; CLIP alone does not supply an
AI-authorship decision. This wrapper loads the trained checkpoint, rather than
comparing the image with text prompts as in zero-shot CLIP.

It uses the same preprocessing and scoring convention as EfficientNet. The
score is the classifier's probability for class `0 = fake`, so higher means
more AI-like according to the fitted classifier. The assumed class order must
match the checkpoint. Requires the internal `bench_imai_artifact` CLIP checkpoint.

### DINOv2 image classifier (supervised baseline)

`DINOv2Detector` uses pretrained, self-supervised DINOv2 image features with a
binary classifier trained on labeled real and generated images. DINOv2 provides
visual representations; the fitted classifier learns which patterns favor the
AI class. This wrapper loads the trained checkpoint.

It uses the same preprocessing and class convention as EfficientNet: the score
is the probability for class `0 = fake`, with higher values meaning more AI-like
according to the fitted classifier. Without that classifier, DINOv2 alone does
not assign authorship labels. This differs from `WaRPADDetector`, which compares
DINOv2 representations without fitting an authorship classifier. Requires the
internal `bench_imai_artifact` DINOv2 checkpoint.

### SigLIP image classifier

`SigLIPDetector` loads a configured Hugging Face image-classification checkpoint,
intended to use SigLIP-2 visual representations fine-tuned for real-versus-fake
classification. Training is supplied by the checkpoint; this wrapper performs
inference through the image-classification pipeline.

The score is the pipeline probability for a label named `fake`, `ai`, or
`synthetic` (checked in that order, ignoring case). Higher means more AI-like
according to that checkpoint. If no recognized label exists, the implementation
returns `0.5`. Labels use a `0.5` threshold, without an `uncertain` band. Requires
the configured model and weights.

### Patch-based classifier

`PatchBasedClassifier` scores local image regions with an existing detector.
It slides `128 x 128` crops with stride `64` over the image, then returns the
**85th percentile** of their scores. This emphasizes high-scoring regions rather
than averaging their evidence away across the whole image.

No additional model is trained. The result depends on the underlying detector,
crop size, and number of patches; improved detection must be validated. If either
image dimension is at or below the patch size, the wrapper scores the whole image
instead. Requires a functioning underlying detector.

### Frequency information and Haar wavelets

`FrequencyDetector` asks whether an image contains an unusual balance of fine
detail and coarse structure. The image is converted to grayscale at native
resolution and a one-level Haar transform separates low-frequency information
(broad color and shape, `LL`) from high-frequency information (edges and
texture, `LH + HL + HH`). The detector measures the **high-to-low energy
ratio** and maps it with a logistic of steepness `20` centered on `0.15`:

$$
S = \frac{1}{1 + \exp\left(-20\left(\frac{E_{LH} + E_{HL} + E_{HH}}{E_{LL}} - 0.15\right)\right)}.
$$

For an illustrative ratio of `0.19`, $S = 0.69$.

**Interpretation:** the sigmoid is extremely steep, so `0.05` maps to `0.12`
and `0.35` maps to `0.98`; the score saturates well before the ratio does. The
center `0.15` is a hand-chosen midpoint, not a fitted threshold and not the
`0.65` label band. The statistic is **relative** high-frequency energy and is
computed at native resolution, so natural texture, resizing, sharpening, and
compression all move it; grayscale conversion discards chromatic artifacts
entirely. This cheap heuristic is **not** WaRPAD and its score is not a
calibrated probability.

### MLEP

`MLEPDetector` searches for **local texture irregularities across image
scales**. It independently shuffles non-overlapping `2 x 2` patches in each
color channel, downscales and restores each of the scales `1`, `0.5`, `0.25`, and
builds exact stride-one `2 x 2` local entropy maps, which are stacked into a
`(255, 255, 9)` map and scaled by `2` into `[0, 1]` before a ResNet-50
classifier.

For an illustrative classifier logit of `1.10`, the score is
`sigmoid(1.10) = 0.75`.

**Interpretation:** a higher output means the multiscale entropy patterns of a
**deliberately shuffled** image resemble generated images in the classifier's
training data. The shuffling happens before the resampling, so the statistic
describes a destroyed local structure, and reproducing a number requires the
same shuffle seed. The permutation and entropy computation do **not** themselves
establish a fake label: the fitted CNN checkpoint is mandatory, its `conv1`
must accept `3 * len(scales)` channels, and the load is strict. Changing `scales`
invalidates a previously trained checkpoint.

### WaRPAD

`WaRPADDetector` tests whether removing fine image detail changes the image's
**high-level representation unusually strongly**. It resizes to `896 x 896`,
removes high-frequency content with a two-level Haar transform
(`x - 0.1 * HF`), tiles the result into `4 x 4 = 16` patches of `224 x 224`, and
compares DINOv2 ViT-L/14 CLS embeddings of the original and perturbed patches.

The score is one minus the mean cosine similarity, clipped to `[0, 1]`.
For an illustrative mean cosine of `0.82`, this gives
`1 - 0.82 = 0.18`.

**Interpretation:** a high cosine indicates representations that are robust to
detail removal, which the method treats as more real-like. There is no sigmoid and no fitted
threshold; with neither a `threshold` nor a `calibrator` supplied, the default
label is `"uncertain"` for **every** image, so the `0.18` above is a usable
number with no decision attached. Supply a threshold and the label is taken
from the raw score; supply a calibrator and `score_semantics` becomes
`calibrated_probability`. Requires DINOv2 weights. One reproduction detail: the
Hugging Face processor resizes each already-`224` patch to `256` and center-crops
back, so the outer margin of every patch never reaches the encoder.

### Denoising trajectory detection

`DenoisingTrajectoryDetector` asks whether real and generated images behave
differently when moved backward and forward along a diffusion process. It
encodes the image into a Stable Diffusion latent using the distribution **mean**
(deterministic), applies DDIM inversion in ascending $t$ over `50` steps,
decodes each predicted-clean latent, and compares CLIP ViT-L/14 layer-`15` CLS
features against the original.

For an illustrative mean cosine similarity of `0.41` over the `50` steps, the
score is

$$
S = \mathrm{clip}\left(\frac{0.41 + 1}{2}, 0, 1\right) = 0.705,
$$

which crosses the `0.65` band in cosine space at a similarity of `0.30`.

**Interpretation:** trajectory-level feature consistency is the detection
signal, but read the direction carefully: the implemented score **increases**
with reconstruction fidelity, so a higher value means the image reconstructs
well through inversion. The returned value is an **uncalibrated similarity
transform** with `score_semantics` set accordingly, and the threshold is a fixed
affine map that was never fitted, so many images land in `uncertain`. Exact
numerical parity with the authors' code and weights has **not** been
established. Cost is 50 UNet forwards plus 50 sequential VAE decodes per image,
and `summed_clip_similarity` in `details` scales with the number of selected
steps, so it is not comparable across `trajectory_indices` settings. Requires
the diffusion and CLIP models.
