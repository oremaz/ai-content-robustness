# AI Content Detector & Evasion Research Framework

A research laboratory for studying the interaction between **AI-content detectors** and **adversarial generators**, across both text and image modalities.

The framework lets you (a) score a piece of text or image with a configurable ensemble of state-of-the-art zero-shot and supervised detectors, (b) train a generator with reinforcement learning to evade those detectors while preserving meaning, and (c) run a multi-round *arms race* where attacker and defender update against each other and the resulting equilibrium is measured.

The detectors and attacks implemented are drawn from the recent literature (ICML 2024, ICLR 2024, NeurIPS 2023, ICLR 2025, ICML 2025, NeurIPS 2025, TMLR 2026). Each component is described in detail below, so you can use this README as a reference without having to read the original papers.

---

## What you can do with it

- **Score text or images** with an ensemble of detectors via a Streamlit UI or programmatic API.
- **Train an evader** (text: GRPO or MultiSPIN on a LoRA adapter; image: DDPO on a Stable Diffusion UNet) that learns to bypass a chosen detector ensemble.
- **Run an arms race**: alternate attacker training and defender retraining over N rounds, log per-round metrics, and read off whether the system converges.
- **Measure adaptation cost** (MAML): how many gradient steps does an attacker need to defeat a brand-new, unseen detector?
- **Compare against published baselines**: vanilla generation, paraphrasing, synonym substitution, prompt engineering, sampling perturbation, OpenRouter API, and the NeurIPS 2025 Adversarial Paraphrasing attack.
- **Explain detector decisions** at the paragraph level using SHAP on a feature-based XGBoost model (notebook).

---

## Repository layout

```
ai_content_detector/
├── app.py                         # Streamlit UI: paste text / upload image, get ensemble scores
├── detectors/                     # Defensive side: text + image detectors and the ensemble logic
├── rl_evasion/
│   ├── run_experiments.py         # Single CLI entry point for every experiment
│   ├── config.py                  # Dataclasses with all hyperparameters
│   ├── text_evasion/              # GRPO, MultiSPIN, HUMPA-style proxy, reward functions, feature bank
│   ├── image_evasion/             # DDPO trainer + diffusion-purification evaluation
│   ├── arms_race/                 # Multi-round attacker/defender loop, RADAR defender, MAML
│   └── benchmarking/              # Datasets, baselines, BenchmarkRunner
├── notebooks/explainable_detector.ipynb   # Paragraph-level XGBoost + SHAP on the MAGE dataset
└── tests/                         # 140+ unit tests pinning the math of every critical component
```

---

## What each file does

### Detectors (`detectors/`)

The defensive side. Every detector implements the same `BaseDetector` interface:

- `is_available() -> bool` — does this machine have everything (model weights, GPU memory) needed?
- `detect(content) -> DetectionResult` — return an AI-probability score in `[0, 1]`, a label, and a per-detector `details` dict.

#### `ensemble.py`
- `BaseDetector` (abstract) and `DetectionResult` (dataclass).
- `EnsembleAggregator` runs every available detector on the same input. It can either average their scores with manual weights or learn the weights from a small calibration set using a logistic-regression meta-classifier. The learned aggregator automatically *down-weights* a detector that is currently being bypassed by an attacker — useful in the arms race loop.

#### `text_detectors.py`

| Class | What it does | Reference |
|---|---|---|
| `ModernBERTDetector` | Supervised classifier: a QLoRA-fine-tuned ModernBERT trained on the MAGE corpus. Outputs the class-0 (machine) probability. | Internal |
| `TFIDFDetector` | TF-IDF features + logistic regression. Cheap CPU-only baseline. Loads from `bench_aitextdetect`. | Internal |
| `BinocularsDetector` | Zero-shot. Loads two small reference LMs (default: Falcon-7B and Falcon-7B-Instruct) and computes the ratio of perplexity (under one model) to cross-perplexity (the same text scored by the other model). Human writing produces a higher ratio than typical LLM samples. | Hans et al., ICML 2024 |
| `FastDetectGPTDetector` | Zero-shot. Computes the *conditional probability curvature* — the gap between the log-probability of the actual text and the expected log-probability of nearby samples drawn from a reference LM. AI text sits in a steeper local maximum. | Bao et al., ICLR 2024 |
| `DivEyeDetector` | Zero-shot. Scores the input under `Qwen/Qwen3.5-9B-Base`, then extracts the distribution of token-level surprisals (mean, std, skewness, kurtosis, first/second derivatives). Human writing has higher surprisal *diversity* than AI. The Base (non-RLHF'd) variant is used on purpose: instruction-tuned models have peaked logits on alignment tokens that distort the diversity signal. | Basani & Chen, TMLR 2026 |
| `IPADDetector` | Faithful reproduction. Uses three published LoRA adapters on `microsoft/Phi-3-medium-128k-instruct`: one *prompt inverter* that hypothesizes the prompt that could have produced the input, and two *distinguishers* (RC and PTCV) that score how well the input is consistent with that hypothesized prompt. Final score = average of the two yes-token probabilities. | Chen et al., NeurIPS 2025 |
| `ParaphraseRoundTripDetector` | Lightweight round-trip heuristic. Asks an LLM to rewrite the input into "clear standard prose"; large normalized edit distance between the original and the rewrite suggests the input is far from the natural-language manifold. Useful as an auxiliary signal, not a stand-alone SOTA detector. The legacy alias `InversionDetector` still works. | DIPPER-inspired, Krishna et al., NeurIPS 2023 |

All zero-shot detectors apply *length-aware thresholds* and a complexity normalizer to mitigate ESL bias (English-as-a-Second-Language writing has lower lexical complexity than native writing and would otherwise be flagged as AI).

#### `image_detectors.py`

| Class | What it does | Reference |
|---|---|---|
| `EfficientNetDetector` | Wraps a fine-tuned EfficientNet-B4-NS (trained on the ArtiFact dataset across 25 generators). | Internal checkpoint |
| `CLIPImageDetector` | Wraps a CLIP-based fake/real classifier. | Internal checkpoint |
| `DINOv2Detector` | Wraps a DINOv2 ViT-B classifier. | Internal checkpoint |
| `SigLIPDetector` | Wraps a HuggingFace SigLIP-based image-classification pipeline that emits a `fake/ai/synthetic` label. | HF model card |
| `PatchBasedClassifier` | Wraps any of the above. Slides a 128×128 window over the image, scores each patch, and aggregates with the 85th percentile. Forces the underlying classifier to detect *local* artifacts rather than relying on a global spurious correlation (lighting, watermark, JPEG fingerprint). | Generic best practice |
| `FrequencyDetector` | Lightweight high-/low-frequency *energy ratio* heuristic on a single-level Haar DWT. Diffusion models leave excess HF energy from the denoising process. Useful baseline; *not* the published WaRPAD algorithm. | — |
| `WaRPADDetector` | Faithful reproduction. For each non-overlapping 224×224 patch over the image (rescaled to 896×896): subtract α·HF(x) where HF is the 2-level Haar high-frequency content (α = 0.1), embed both the original and the perturbed patch with **DINOv3 ViT-L/16** (`facebook/dinov3-vitl16-pretrain-lvd1689m`), take the cosine similarity of their CLS tokens. Real photos give embeddings that are *robust* to HF perturbations (cosine close to 1); generated images do not. (Note: DINOv3 weights are gated on HuggingFace — run `huggingface-cli login` once.) | Choi et al., NeurIPS 2025 |
| `DenoisingTrajectoryDetector` | DTAD-style. Encodes the input image to a Stable Diffusion VAE latent. At a grid of timesteps, adds Gaussian noise to the latent and runs a single-step DDIM denoise to predict `x_0`. Generated images converge faster along this trajectory than real photos, so the average cosine similarity between predicted `x_0` and the original latent is higher for AI images. | Liu et al., NeurIPS 2025 (DTAD); implementation grounded in LATTE (Vasilcoiu et al.) |

### Reinforcement-learning evasion (`rl_evasion/`)

The offensive side. The CLI entry point is `rl_evasion/run_experiments.py`.

#### Text (`rl_evasion/text_evasion/`)

- **`grpo_trainer.py`** — Group Relative Policy Optimization. For each prompt, sample K completions with the current policy, score them with the reward function, compute group-normalized advantages, and update the policy via per-token policy gradient + a KL penalty against a frozen reference (the same model with its LoRA adapter disabled). Uses TRL's `GRPOTrainer` when available and falls back to a built-in reference loop otherwise.
- **`multispin.py`** — MultiSPIN. Iterative self-play: at each step the model generates its own response to a prompt and is updated with the DPO-style SPIN log-sigmoid loss to prefer the human reference over its own generation. Stylometric features (burstiness, type-token ratio, sentence-length variance, function-word ratio, POS bigram entropy, hapax ratio) and embedding-MMD distances are computed for monitoring; the SPIN loss is the actual gradient signal.
- **`proxy_evasion.py`** — A *research stub* exploring the decoding-time logit-shift idea (HUMPA, Wang et al., ICLR 2025). Adds the logits of a small "humanizing" proxy model to the logits of the target model at every step. The proxy training loop is not implemented — treat this as a hook for experimentation, not a finished method.
- **`feature_bank.py`** — Persistent memory of probe classifiers. After a training round, train a small linear probe on stylometric features that separate the latest generations from human text; keep the K probes with highest AUROC. Future rounds can include the bank's confidence as an extra reward term so the attacker has to defeat *every* feature family it has previously failed on.
- **`rewards.py`** — Reward functions: detector-evasion (`1 − detector_score`), semantic similarity (E5 sentence embeddings, mean-pooled with the `query:` prefix), and a quadratic length-ratio quality penalty. `CompositeReward` combines them and rejects degenerate short outputs (default: < 20 tokens) by returning `total = 0`.
- **`evaluate.py`** — Evaluation harness: aggregates per-detector evasion, attack-success rate, mean semantic similarity, and TPR at given FPR operating points (linearly interpolated, not nearest-neighbor).

#### Image (`rl_evasion/image_evasion/`)

- **`ddpo_trainer.py`** — Denoising Diffusion Policy Optimization. Treats every DDIM denoising step as an RL action and updates the UNet (with LoRA) via importance-sampled policy gradient with PPO-style ratio clipping. Reward = detector-evasion + CLIP image-text alignment + aesthetic score. Includes evaluation under **diffusion purification** (a separate, *pretrained* `StableDiffusionImg2ImgPipeline` adds noise at strength 0.15 and re-denoises the attacker's output before scoring; this measures whether the evasion survives spectral scrubbing). | Black et al., NeurIPS 2023; Saberi et al., ICML 2025

#### Arms race (`rl_evasion/arms_race/`)

- **`equilibrium.py`** — The headline experiment. For N rounds: (1) the attacker fine-tunes its policy against the current detector ensemble; (2) generates fresh samples on a *training* prompt subset; (3) the defender retrains on those samples; (4) the attacker's reward function is updated with the retrained defender. All evaluation is on a *disjoint* held-out prompt subset (configured at setup time, fixed seed). The per-round **Nash gap** is `max(0, defender_accuracy − (1 − attacker_success_rate))`, bounded in `[0, 1]`. A gap that converges to a small constant indicates equilibrium; a widening gap means the defender is winning.
- **`radar_defender.py`** — RADAR-style adaptive defender. Retrains a RoBERTa classifier on the latest attacker outputs vs. matched human references after each round. | Hu et al., NeurIPS 2023
- **`meta_adapt.py`** — MAML-style meta-learning over a "detector zoo". The inner loop adapts the policy on a *support* prompt subset against a sampled detector; the outer loop minimizes the post-adaptation loss on a disjoint *query* subset. Uses `torch.func.functional_call` for proper second-order gradients when available, falling back to first-order MAML otherwise. The companion method `measure_adaptation_cost` evaluates how many gradient steps are needed to reach an evasion threshold against a brand-new, held-out detector. | Finn et al., ICML 2017

#### Benchmarking (`rl_evasion/benchmarking/`)

- **`datasets.py`** — Dataset loaders: HC3 (human/AI Q&A pairs), CNN/DailyMail (prompt-only), and a generic file loader. Each dataset exposes `ai_texts_available` so the runner knows whether post-hoc methods (paraphrase, synonym substitution) are usable on it.
- **`baselines.py`** — Six evasion baselines:
  - `VanillaBaseline` — no transformation (lower bound).
  - `ParaphrasingBaseline` — Pegasus-paraphrase, sentence by sentence.
  - `SynonymSubstitutionBaseline` — WordNet content-word swaps.
  - `PromptEngineeringBaseline` — prepend a "write like a human" instruction.
  - `SamplingPerturbationBaseline` — high temperature + nucleus sampling.
  - `OpenRouterBaseline` — paraphrase via any model on OpenRouter (requires `OPENROUTER_API_KEY`).
  - `AdversarialParaphrasingBaseline` — the strongest training-free 2025 humanizing attack: iteratively paraphrase with `Qwen/Qwen3.5-9B` (Instruct), score every candidate with a *guidance detector*, and accept the best (lowest AI-score) until either the threshold τ is reached or the iteration cap is hit. | Chen et al., NeurIPS 2025
- **`benchmark.py`** — `BenchmarkRunner`. Runs every method on a dataset and reports per-detector evasion + a held-out evasion (computed against detectors that were *not* in the training set, passed via `heldout_detectors=`). It refuses to construct a runner where the held-out and training detectors overlap — this prevents the benchmark from over-crediting methods that already optimized against the eval pool.

### `app.py`
A Streamlit UI. The text tab takes a paragraph and shows each available detector's score plus the ensemble verdict. The image tab does the same for an uploaded image. Detectors that can't be loaded on the host machine are listed in an "Unavailable detectors" panel rather than crashing the page.

### `notebooks/explainable_detector.ipynb`
Trains a paragraph-level XGBoost classifier on the MAGE dataset using the surprisal + stylometric features defined in `multispin.py`'s extractors. Then uses SHAP `TreeExplainer` to visualize, per paragraph, which features pushed the model toward "AI" or "human". Document-level macro-averaged AUROC is reported alongside paragraph-level AUROC, with disjoint document IDs across train and test asserted at split time so paragraph correlation can't leak across the boundary.

### `tests/`
146 unit tests pinning the math of every critical component: reward sign convention and length penalty, GRPO advantage normalization and policy-gradient sign, MultiSPIN DPO log-sigmoid arithmetic, the WaRPAD Haar HF reconstruction (constant image → zero HF; noise → non-zero; offset edge → localized HF), the equilibrium Nash-gap bounds and monotonicity, ensemble aggregation, feature-bank probe training, evaluation metrics, and image-detector return-type contracts. The tests are pure-Python and CPU-only — they don't require any LLM or diffusion model to be installed.

---

## Hardware requirements

The framework is designed to degrade gracefully: anything that won't fit on the available hardware is reported as `is_available() == False` and the rest keeps working.

### Minimum (CPU only, ~8 GB RAM)
- Streamlit app with the lightweight detectors: `TFIDFDetector`, `FrequencyDetector`, `ParaphraseRoundTripDetector` (CPU-slow), and the supervised checkpoint detectors that fit on CPU (`ModernBERTDetector` in QLoRA mode, classifier-only image detectors).
- The full test suite (pure-Python, no model loads).
- Non-RL baselines: `VanillaBaseline`, `SynonymSubstitutionBaseline`, `OpenRouterBaseline`.
- ❗ **Not** available on CPU: `DivEyeDetector` (now scored with Qwen 3.5 9B-Base, ~18 GB bf16), Binoculars / Fast-DetectGPT (Falcon-7B pair), `WaRPADDetector` (DINOv3 ViT-L/16 needs CUDA in practice), `IPADDetector`, all RL trainers (GRPO/MultiSPIN/DDPO).

### Mid-range (single GPU, 24 GB VRAM — e.g. RTX 4090, A10G, L4)
Sufficient for:
- `DivEyeDetector` with `Qwen/Qwen3.5-9B-Base` in bf16 (~18 GB resident).
- `BinocularsDetector` and `FastDetectGPTDetector` with the Falcon-7B pair in 4-bit quantization.
- All image detectors **except** `IPADDetector` and full-resolution DenoisingTrajectoryDetector + DDPO at the same time: `EfficientNetDetector`, `CLIPImageDetector`, `DINOv2Detector`, `SigLIPDetector`, `WaRPADDetector` (DINOv3 ViT-L/16 in fp16 ≈ 1.2 GB), `FrequencyDetector`, `DenoisingTrajectoryDetector` (Stable Diffusion v1.5 in fp16 ≈ 4 GB) one at a time.
- DDPO image evasion with LoRA on Stable Diffusion v1.5 (≈ 12 GB during training).
- GRPO / MultiSPIN with `Qwen/Qwen3.5-9B-Base` + LoRA (≈ 18 GB) — keep `grpo_num_generations=4` and `per_device_train_batch_size=2`.
- `AdversarialParaphrasingBaseline` with `Qwen/Qwen3.5-9B` (Instruct) — ~18 GB.
- The arms-race loop in text mode for short runs (3–5 rounds).

### High-end (single GPU ≥ 40 GB VRAM, e.g. A100-40/80 GB, H100)
Adds:
- `IPADDetector` — ~28 GB for `microsoft/Phi-3-medium-128k-instruct` in bf16 plus the three published LoRA adapters (`bellafc/IPAD/Prompt_Inverter`, `Distinguisher_RC`, `Distinguisher_PTCV`). This is the only Phi-3 holdout in the project and exists because LoRA adapters are tied to their training base; swapping to Qwen would silently break the published weights.
- Binoculars / Fast-DetectGPT in fp16 without quantization.
- DDPO training + diffusion-purification evaluation simultaneously (two SD pipelines resident).
- The full arms race for 10+ rounds with frequent defender retraining.
- Running the entire ensemble (every detector simultaneously) for benchmark-table generation.

### Default model footprint (bf16 unless noted)

| Component | Model | ≈ VRAM |
|---|---|---|
| Binoculars / Fast-DetectGPT | `tiiuae/falcon-7b` ×2 | 28 GB fp16 / 8 GB 4-bit |
| DivEye scoring | `Qwen/Qwen3.5-9B-Base` | 18 GB |
| IPAD | `microsoft/Phi-3-medium-128k-instruct` + 3 LoRA | 28 GB |
| ParaphraseRoundTrip | `Qwen/Qwen3.5-4B` | 8 GB |
| Style detector | `rrivera1849/LUAR-MUD` | < 1 GB |
| Semantic-similarity reward | `intfloat/e5-base-v2` | < 1 GB |
| WaRPAD backbone | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1.2 GB fp16 |
| DenoisingTrajectory | `stable-diffusion-v1-5/stable-diffusion-v1-5` | 4 GB fp16 |
| GRPO / MultiSPIN base | `Qwen/Qwen3.5-9B-Base` + LoRA | 18 GB + ~2 GB activations |
| Adversarial Paraphrasing | `Qwen/Qwen3.5-9B` (Instruct) | 18 GB |
| RADAR defender | `roberta-base` | < 1 GB |
| DDPO base + LoRA | Stable Diffusion v1.5 | 12 GB during training |

If a model needs more memory than is available, the corresponding `is_available()` returns False and that detector / trainer is skipped without affecting the rest of the run.

---

## Step-by-step guide

### 1. Install dependencies
```bash
# From the repo root, in your venv (uv, conda, or plain venv all fine)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
The `requirements.txt` pins `transformers`, `peft`, `diffusers`, `trl`, `torch`, `pywavelets`, `editdistance`, `xgboost`, `shap`, `streamlit`, and the dataset packages.

### 2. Verify the install
```bash
PYTHONPATH=. pytest ai_content_detector/tests/ -v
```
Expected: 146 tests pass in ~10 seconds (CPU only).

### 3. Smoke-test the wiring
```bash
PYTHONPATH=. python -m ai_content_detector.rl_evasion.run_experiments --experiment smoke_test
```
Imports every component, instantiates the lightweight ones, and prints a per-stage status. This is the fastest way to confirm a fresh checkout is healthy.

### 4. Try the interactive app
```bash
streamlit run ai_content_detector/app.py
```
- The text tab runs every available text detector on whatever paragraph you paste and shows per-detector scores + an aggregate verdict.
- The image tab does the same for an uploaded image.
- An "Unavailable detectors" panel lists what couldn't be loaded (missing checkpoint, insufficient VRAM, etc.) so you know what's running.

### 5. Run a single evasion experiment

The CLI is `python -m ai_content_detector.rl_evasion.run_experiments --experiment <name>`. The available experiments:

| Experiment | What it does | Typical runtime (1× A100) |
|---|---|---|
| `smoke_test` | Imports / instantiation only. No training. | ~30 s |
| `grpo_text` | GRPO on a 7–9 B causal LM with LoRA against the default text detector ensemble. | 1–4 h depending on `--epochs` |
| `multispin` | MultiSPIN distribution matching, 5 iterations. | 2–6 h |
| `ddpo_image` | DDPO on Stable Diffusion v1.5 LoRA against the default image detector ensemble. Includes diffusion-purification evaluation at the end. | 3–8 h |
| `arms_race --modality text` | 10-round attacker (GRPO) ↔ defender (RADAR retraining) loop with disjoint train/eval prompts. | 6–24 h |
| `arms_race --modality image` | Same loop, image side. Defender is a placeholder. | 6–24 h |
| `meta_adapt` | MAML over a detector zoo, then `measure_adaptation_cost` against held-out detectors. | 4–12 h |
| `benchmark` | Runs every baseline + your trained checkpoints on HC3 and reports a comparison table with held-out evasion. | 30 min – 2 h |

Useful flags:
- `--model <hf_repo>` — override the default base model.
- `--epochs N` — override training epochs.
- `--rounds N` — override arms-race rounds.
- `--output-dir path/` — where checkpoints and per-round JSON history are written.

Example:
```bash
PYTHONPATH=. python -m ai_content_detector.rl_evasion.run_experiments \
    --experiment grpo_text \
    --model Qwen/Qwen3.5-9B-Base \
    --epochs 2 \
    --output-dir results/grpo_qwen
```

### 6. Run the explainability notebook
```bash
jupyter notebook ai_content_detector/notebooks/explainable_detector.ipynb
```
Loads MAGE, splits paragraphs from disjoint documents, extracts surprisal + stylometric features, fits an XGBoost classifier, and shows SHAP waterfall + summary plots. Both paragraph-level and document-level macro-averaged metrics are printed.

### 7. Reproduce the headline comparison
```bash
PYTHONPATH=. python -m ai_content_detector.rl_evasion.run_experiments \
    --experiment benchmark \
    --output-dir results/benchmark
```
Produces a JSON file with one row per method (RL methods + the seven non-RL baselines, including `AdversarialParaphrasingBaseline`) and one column per detector + a held-out evasion column. The held-out column is the right number to read for the "did our RL attack actually generalize?" question.

---

## Reproduction caveats

A short list of where each component sits relative to the original paper, so reviewers know what they're looking at:

- **IPAD** — uses the authors' three released LoRA adapters on `microsoft/Phi-3-medium-128k-instruct`; our contribution is the ensemble integration.
- **WaRPAD** — exact algorithm (2-level Haar HF + DINOv2-L/14 cosine sensitivity + non-overlapping 224² patches over an 896² rescale, α = 0.1), hyperparameters as in the paper.
- **DenoisingTrajectoryDetector** — DTAD-style metric; since DTAD itself has no public code, the implementation is grounded in the released LATTE pipeline (Vasilcoiu et al.).
- **Adversarial Paraphrasing** — port of the authors' algorithm with their default hyperparameters; the paraphraser is configurable (default Llama-3.1-8B-Instruct).
- **HUMPA proxy attack** — `proxy_evasion.py` is a research stub demonstrating the decoding-time logit-shift idea; the proxy training loop is not implemented.
- **GRPO** — the manual fallback path is a minimal reference implementation (group-normalized advantages, per-token policy gradient, k2 KL to a frozen reference). For production runs, prefer the TRL path.
- **MultiSPIN** — applies the DPO-style SPIN log-sigmoid loss; the stylometric / embedding / style-MMD distances are logged as monitoring metrics. They don't carry gradients here because text decoding is non-differentiable.
- **Arms-race Nash gap** — best-response deficit on a held-out prompt split. Empirical proxy, not a theoretical equilibrium guarantee.
- **MAML adaptation cost** — the inner loop uses `torch.func.functional_call` for proper second-order gradients when available, otherwise first-order MAML. The loss is an evasion-weighted LM surrogate.
- **`ParaphraseRoundTripDetector`** — DIPPER-style round-trip heuristic, not a SOTA detector on its own.

---

## Key references

**Detection**
- Hans et al., *Spotting LLMs With Binoculars* (ICML 2024)
- Bao et al., *Fast-DetectGPT* (ICLR 2024)
- Basani & Chen, *Diversity Boosts AI-Generated Text Detection* (TMLR 2026)
- Chen et al., *IPAD: Inverse Prompt for AI Detection* (NeurIPS 2025)
- Choi et al., *WaRPAD: Training-free Detection via Cropping Robustness* (NeurIPS 2025)
- Liu et al., *Denoising Trajectory Biases for Zero-Shot AI-Generated Image Detection* (NeurIPS 2025)
- Krishna et al., *Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense* — DIPPER (NeurIPS 2023)
- Hu et al., *RADAR* (NeurIPS 2023)

**Evasion**
- Chen et al., *Self-Play Fine-Tuning Converts Weak LMs to Strong LMs* — SPIN (ICML 2024)
- Wang et al., *Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors* — HUMPA (ICLR 2025)
- Chen et al., *Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text* (NeurIPS 2025)
- Saberi et al., *Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks* — Diffusion Purification (ICML 2025)
- Black et al., *Training Diffusion Models with RL* — DDPO (NeurIPS 2023)
- DeepSeekMath team, *GRPO* (2024)
- Finn et al., *Model-Agnostic Meta-Learning* (ICML 2017)
