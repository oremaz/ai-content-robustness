# Using the authors' DivEye classifier

`DivEyeDetector` now loads the authors' pinned pretrained classifier and its
matching **83-feature DivEye/BiScope pipeline**. It replaces the local nine-
feature implementation. No classifier training is needed for initial inference.
Falcon-7B and Gemma-1.1-2B-Instruct are placed on the same requested device,
put into evaluation mode, and have gradients disabled. Attacker RL does not
update their weights or the XGBoost classifier.

```python
from ai_content_detector.detectors.text_detectors import DivEyeDetector

detector = DivEyeDetector(device="cuda:0")
result = detector.detect("A passage of at least fifteen words is required by the released detector before it can be scored.")
```

The release revision is `d2961b844705c0aa6f95ded41dc24ba18d2d445f`.
The default classifier download must match the checksum in the inspection receipt.
Falcon is pinned to `ec89142b67d748a1865ea4451372db8313ada0d8` and Gemma to
`d750f5eceb83e978c09e2b3597c2a8784e381022`. These identify our extractor
weights; the original application names the model repositories without pinning
revisions, so its historical runtime cannot be reconstructed from app.py alone.
The source Space declares `cc-by-nc-sa-4.0`; attribution is to Advik Raj Basani
and Pin-Yu Chen, with the pinned source links below.

The implementation reproduces the application feature order, population
variance conventions, decile suffix statistics, token offsets and AI column 1.
Undefined moments remain NaN, which the released XGBoost model treats as missing
features. It rejects incompatible feature counts and inputs below 15 words.
Both LM forward passes use inference mode and disable unused generation caches.
CUDA uses the release's FP16; CPU uses FP32. Device placement and inference
housekeeping differ from the two-GPU Gradio application. Numerical agreement
on real texts has not yet been measured.

The classifier JSON has 83 features and classes `[0, 1]`. The pinned model
weights and tokenizers are cached for execution. The ensemble starts from the
saved warmed classifier; its Falcon and Gemma extractors remain frozen during
this defender update.
A local authorized copy can be supplied with `model_paths={"bi": "/path/to/checkpoint"}`;
its path is recorded in detection details. Supplying a different Gemma variant
would invalidate the classifier's feature compatibility.

The investigation below explains why downloading only the classifier into the
previous extractor was insufficient. It contains no measured DivEye accuracy
or attack robustness results.

## What the paper trains

The full 41-page TMLR paper, arXiv `2509.18880v3`, was downloaded and read.
Its feature extractor freezes an autoregressive language model, usually GPT-2.
For token \(x_t\), surprisal is
\(s_t=-\log p(x_t\mid x_{<t})\). Nine statistics summarize the sequence:
four distribution moments, the mean and variance of first differences, and
the variance, histogram entropy, and autocorrelation of second differences.
An XGBoost classifier then learns from labeled human and AI examples.
The extractor requires no fine-tuning; the final classifier requires
supervised training. Algorithm 1 uses `0=AI` and `1=human`.
[Paper, Sections 3 and 4 and Appendix C](https://arxiv.org/pdf/2509.18880v3).

Appendix I specifies depth 12, 200 estimators, column sampling 0.8,
row sampling 0.7, minimum child weight 5, gamma 1, seed 42, and
`scale_pos_weight = count(label=0) / count(label=1)`.
Entropy uses 20 bins and tokenization truncates at 1,024 tokens.
MAGE OOD testbeds reuse the classifier trained on Testbed 4.
[Paper, Appendices H and I](https://arxiv.org/pdf/2509.18880v3).

## A downloadable artifact exists, but its schema differs

The authors' linked Hugging Face Space publishes `model.json`. Inspection of
the downloaded XGBoost JSON found `num_feature=83`. Its application constructs
11 features from Falcon-7B surprisal and compression, followed by 72 BiScope
features from Gemma-1.1-2B-Instruct. The application takes probability column 1
as AI. This is a combined detector with a different feature schema and label
orientation from the previous standalone implementation.
[Pinned Space application](https://huggingface.co/spaces/pinyuchen/Diveye_AI_text_detector/blob/d2961b844705c0aa6f95ded41dc24ba18d2d445f/app.py),
[pinned classifier](https://huggingface.co/spaces/pinyuchen/Diveye_AI_text_detector/blob/d2961b844705c0aa6f95ded41dc24ba18d2d445f/model.json).

The artifact can be fetched explicitly with:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="pinyuchen/Diveye_AI_text_detector",
    repo_type="space",
    filename="model.json",
    revision="d2961b844705c0aa6f95ded41dc24ba18d2d445f",
)
```

The replacement implementation loads this native XGBoost JSON directly,
using the exact 11 + 72 feature schema. The previous joblib/nine-feature path
has been removed. An explicitly supplied `classifier_path` must also be a
binary 83-feature native XGBoost checkpoint; the caller is responsible for
preserving its feature order and label meaning. A warm-up checkpoint must be
kept separate from the authors' original classifier.

## Distinguishing the standalone paper training route

The official repository accepts CSV files with `text` and `label` columns:

```bash
python diveye.py --model gpt2 \
  --train_dataset train.csv --test_dataset test.csv
```

At the inspected revision, its utility returns ten features, including two
standard deviations, rather than the paper's nine. It also uses population
variances and a different autocorrelation calculation. Therefore its trained
classifier cannot be transplanted into this repository without aligning the
extractor. The CLI trains and evaluates but does not save the classifier.
[Official training code](https://github.com/IBM/diveye/blob/ee7d53abfb56ea4dc4cac2cb24e17beec0146e5e/diveye.py),
[official feature utility](https://github.com/IBM/diveye/blob/ee7d53abfb56ea4dc4cac2cb24e17beec0146e5e/diveye_utils.py).

The standalone paper protocol is not substituted for the downloaded combined
classifier. Future Qwen-specific warm-up of the released classifier would keep
its frozen Falcon/Gemma feature extractors and continue supervised XGBoost
training on separate training examples. That adaptation is implemented as the distinct Qwen warm-up stage,
not part of the unmodified released detector. Its training IDs,
feature configuration, label orientation, checkpoint hash and validation choices
must be recorded. Benchmark test examples and evaded evaluation outputs must
stay outside classifier fitting and threshold selection.

## Inspection receipt

Inspected on 2026-10-14:

| Artifact | Immutable identity |
|---|---|
| Full PDF | SHA256 `c0f0f6366a8958c2381f57a1bfed9e66de649f90eb3928d7f45af6a1534ddbb6` |
| Official repository | `ee7d53abfb56ea4dc4cac2cb24e17beec0146e5e` |
| Authors' Space | `d2961b844705c0aa6f95ded41dc24ba18d2d445f` |
| Space classifier JSON | SHA256 `ee1fce303de8b44777b608645ea1bce8c92c55e2a776d80d5662f744b8b64ac7` |
