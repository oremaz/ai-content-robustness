# Watermarking and removal experiments

The implementation studies four families:

The removal protocol treats the watermark technology as unknown to the
attacker. The [literature review and shortlist](watermark_removal_literature.md)
select two model-level families: clean-checkpoint merging when clean weights
are available, and independent clean-data training when they are unavailable.
Owner detectors are reserved for separate evaluation; generic detection is not
assumed to certify removal of arbitrary unknown schemes.

`scripts/watermark_removal.py` implements this protocol using the existing clean
training loops. Each worker receives a validated task with an opaque model ID,
native architecture, immutable model/data inputs and a fixed recipe. Only merge
tasks receive clean reference weights. Owner manifests, keys, payloads, detector
weights and scores are not worker arguments. Owner evaluation is a separate
process and waits for all candidate artifact receipts and hashes.

The fixed candidates are merge fractions 0.75, 0.5, 0.25 and full replacement,
plus clean LoRA and full-parameter training checkpoints at 500 and 2,500 steps.
All candidates are reported; owner scores never select a checkpoint. Full Qwen
training keeps parameters and optimizer states in float32, using bfloat16
autocast for forward computation. Its GPU preflight exercises two optimizer
steps with the actual accumulation and context length before long training.

Image training uses 8,000 clean ImageNet training images from 800 classes and
800 separate images for validation. Classes 800 through 999 are excluded from
attack fitting; owner evaluation reports these separately. Labels follow the
original fixed synset order. The ordinary generator backbone is the predefined
training target; the decoder is unchanged without consulting watermark location.
Text fitting and validation are split by question ID within the original removal
partition, with no owner training or test documents supplied to the worker.
These text data cover HC3 questions and do not establish removal across languages
or arbitrary domains. Image quality uses CLIP alignment rather than 50K-image FID.

```bash
PYTHONPATH=. uv run --no-sync python scripts/watermark_removal.py prepare \
  IMAGE_WATERMARK_RUN/manifest.json LATENT_TEXT_RUN/manifest.json
PYTHONPATH=. uv run --no-sync python scripts/watermark_removal.py submit NEW/manifest.json
```

Submission holds jobs for inspection before release. Each model has a serialized
probe, merge, LoRA, full-training and owner-evaluation lane on regular H100 QoS.
The lane waits for its watermarked model producer and for the previous baseline
removal lane to finish, preserving existing jobs and limiting concurrency to
three GPUs. Completed producer receipts can satisfy prerequisites after Slurm
has purged the completed job ID. No-clean-weights methods do not become eligible
to use the clean checkpoint merely because another method can use it.

Input separation is enforced by task schemas, separate process arguments and
artifact hashes; it is not an operating-system sandbox against a malicious
worker inspecting the shared account's filesystem. Tests cover private-field
rejection, train/validation separation, merge endpoints, artifact integrity,
owner-evaluation gating, Slurm arguments and native RAR LoRA/full backward and
checkpoint reload. The original sampler-only TextSeal results remain a separate
control and are not relabeled as model-weight removal.

| Family | Watermark location | With original weights | Without original weights |
| --- | --- | --- | --- |
| Official TextSeal + Qwen3.5-4B-Base | Keyed dual-key sampler | Ordinary sampling | The same ordinary-sampling bypass, using the released model |
| DistSeal-inspired latent Qwen3.5-4B-Base | Learned hidden-state teacher, distilled into merged weights | Interpolation with the original model, including replacement | Clean text fine-tuning of the merged watermarked model |
| Official DistSeal + RAR-XL | Autoregressive generator weights | Generator interpolation and replacement | LoRA fine-tuning on clean image tokens |
| Official DistSeal + UViT-H/DC-AE | Diffusion generator weights | Generator interpolation and replacement | LoRA fine-tuning on clean image latents |

The image experiments load released generator checkpoints and their corresponding
unmodified decoders and extractors. They do not retrain DistSeal. Original weights
mean the pre-watermark generator; unchanged decoder components of the released
watermarked pipeline remain available in both settings. Detector keys and outputs
are used for evaluation, and never enter the removal loss.

## Relationship between the methods

[DistSeal](https://proceedings.mlr.press/v306/rebuffi26a.html) trains a latent embedder/extractor and
then trains a generator on its watermarked latents, or trains a decoder to
reconstruct watermarked images. Appendix G of the ICML 2026 version demonstrates forgetting during
clean LoRA fine-tuning, using 50 images of one ImageNet class. The experiment here
uses that generator setting, with 2,500 removal steps at learning rate 1e-4.
Evaluation includes both that class and other classes.

[TextSeal](https://arxiv.org/abs/2605.12456) modifies sampling rather than model
weights. It uses two keyed Gumbel samplers, entropy-aware scores and localized
detection. The paper also demonstrates watermark transfer through training on
sampled teacher outputs, evaluated using a pooled teacher-forcing radioactivity
test. That is different from detecting a watermark in each fresh student response.
Our TextSeal evaluation uses the corrected localized ensemble, a three-token
context and 90% routing to key A. A small adapter constructs a configured
Transformers `DynamicCache` for Qwen's hybrid layers; the upstream fallback uses
an incompatible cache with this installed Transformers version.

The original green-list KL branch was canceled before training. It was
logit-based watermark distillation, as studied by
[Gu et al.](https://arxiv.org/abs/2312.04469), and did not implement DistSeal's
learned latent watermarking stage. Its four Slurm jobs remain as canceled
receipts; the original TextSeal and image jobs are preserved.

The replacement learns a message-conditioned residual embedder on Qwen's final
normalized hidden states, before its frozen vocabulary head. A separate learned
extractor receives only decoded and retokenized output text and predicts 32
message bits. Its token embedding, two convolution layers and masked pooling
need no Qwen states at detection time. Message BCE trains both networks jointly.
A forward KL quality constraint targets 0.1 nats per token; a dual penalty
controls its weight. Each latent coordinate changes by at most 0.35 times the
hidden-state RMS. Opposite random messages share each training context,
preventing context-to-message label leakage.

Discrete text requires an additional approximation: the forward loss uses
sampled, decoded and retokenized hard text, while a straight-through gradient
through expected token embeddings updates the embedder. Extractor gradients
come only from hard text. This biased continuous surrogate is specific to this
experimental adaptation. DistSeal's image decoder and discriminator are
replaced by the frozen vocabulary head, discrete text channel and KL constraint;
this is not an official text implementation or a reproduction of its image
training objective.

Joint training runs for 6,000 steps with two contexts and their paired messages
per step, 256 prediction positions and width-128 networks. Every 250 steps,
selection measures hard-text bit recovery on disjoint validation contexts,
subject to KL at most 0.125 nats per token. The selected teacher must then pass
a free-running, 64-prompt validation probe: bit accuracy at least 60%, AUROC
at least 0.7 and detection at least 25% at its validation threshold. Failure
blocks the expensive student jobs and is reported explicitly.

After this gate, a fixed registered message conditions the learned teacher.
Dense `KL(q_learned || p_student)` trains Qwen LoRA rank 64 for 5,000 optimizer
steps, eight sequences per step, maximum length 512, learning rate 3e-5.
Vocabulary KL is summed and averaged over valid prediction positions.
Adapters are merged; ordinary student inference uses neither an embedder hook
nor an external watermarking sampler. Teacher and student use temperature 0.8
and full-support sampling (`top_p=1`). The payload cannot change at inference.

### What the detector knows

The owner tokenizes candidate text, applies the learned extractor, and scores
agreement of its bit probabilities with the registered message. Responses
shorter than 32 tokens are rejected. For eligible texts, the threshold is the
larger of the separate 99th percentiles on human and ordinary-base
validation responses. Tests report human and base-model false positives
separately, with confidence intervals. Strong detection by this informed
detector is intended; it does not establish outsider detectability or removal
resistance.

The 32-bit message is an identifier, not a cryptographic key. Random-message
controls check specificity using the frozen owner threshold; they do not
estimate a calibrated per-key false-positive probability. A character
TF-IDF/logistic classifier learns from 128 paired base/student training
responses and tests disjoint responses, without extractor or message access.
Its AUROC probes outsider distinguishability; a weak result would not prove
security. Random token deletion provides an output-level diagnostic.
Deletion is applied to both marked and negative texts; the clean owner
threshold stays fixed, so changed false positives remain visible.
Paraphrase and forgery resistance remain unestablished by these jobs.

The distribution-preservation caveat concerns what is averaged. With a fixed key,
TextSeal creates conditional token preferences that can be learned; averaging
those preferences over keys recovers the base distribution under the paper's
assumptions. Distilling that averaged base distribution supplies no watermark
signal. The learned message-conditioned teacher avoids that ambiguity. It does not
claim distortion-free outputs or arbitrary message selection at inference.

## Removal literature and interpretation

[Towards Watermarking of Open-Source LLMs](https://arxiv.org/abs/2502.10525)
systematically evaluates merging, clean fine-tuning, pruning and quantization.
Its results motivate testing weight changes while measuring model utility,
and evaluating the domain targeted by fine-tuning. Our bounded experiments use
linear interpolation, rather than reproducing that paper's SLERP implementation.

Meta's [Stable Signature](https://arxiv.org/abs/2303.15435) embeds watermarks in
the latent decoder and studies reconstruction-based purification and collusion.
[Stable Signature is Unstable](https://arxiv.org/abs/2405.07145) further studies
decoder removal with and without encoder access, combining reconstruction,
perceptual and adversarial losses. Those decoder attacks are relevant context;
the jobs here target DistSeal's watermarked generators using their native clean
training objectives. They do not implement that decoder attack as if it were a
DistSeal reproduction.

Replacing a model with its exact original trivially removes changes introduced
by watermark training. Intermediate interpolation points measure whether removal
also succeeds while retaining some modified weights. Withholding an original
copy does not establish security: clean fine-tuning may still erase a watermark.
Published image checkpoint messages are public, so these runs test removal rather
than secret-key security.

## Data, measurements and jobs

HC3 human documents are split by normalized question hash into disjoint training,
removal, validation and test groups. Only documents with at least 320 Qwen tokens
are used. Generation uses the first 50 tokens as the prompt and up to 256 response
tokens. The fixed test set contains 256 documents. Evaluation reports empirical
watermark detection and false positives at a threshold chosen on independent
validation documents, confidence intervals, AUROC, clean held-out
perplexity, output length and repetition. Generation perplexity is measured by the
original Qwen as an independent evaluator after removal; that evaluator is never
supplied to the optimizer in the setting without original weights. The original
model and external learned latent teacher are controls for the distilled
student. TextSeal also tests dilution.

Image removal uses 50 ImageNet training images from class zero. Generated
evaluation uses 512 images, including 128 from that class; calibration uses
256 different base-model noise seeds. Metrics include bit accuracy, detection,
false positives, confidence intervals, AUROC and CLIP class alignment. JPEG 50
and a center crop retaining half the area are evaluated with thresholds frozen
from correspondingly transformed calibration images. These sample counts cannot
validate extremely small false-positive rates or reproduce the paper's
50,000-image FID results.

All stages use regular H100 QoS, immutable source snapshots, pinned upstream
revisions and checkpoint hashes. A short GPU preflight gates generation and
training. Three dependency lanes limit the experiment to three simultaneous
GPUs. Training saves restart state and requeues on Slurm's advance signal.

```bash
# Preserve the installed PyTorch and Transformers stack.
MAX_JOBS=4 uv pip install --no-deps -r requirements_watermarking.in
PYTHONPATH=. uv run --no-sync python scripts/watermark_experiments.py resources
PYTHONPATH=. uv run --no-sync python scripts/watermark_experiments.py prepare
PYTHONPATH=. uv run --no-sync python scripts/watermark_experiments.py submit PATH/manifest.json

# New frozen run reusing the parent's immutable resources and partitions.
PYTHONPATH=. uv run --no-sync python scripts/watermark_experiments.py adapt-text PARENT/manifest.json
PYTHONPATH=. uv run --no-sync python scripts/watermark_experiments.py submit NEW/manifest.json
```

Submission initially holds the jobs for allocation and dependency inspection.
Release the recorded IDs with `scontrol release`. Each run maintains `report.md`
and individual JSON measurements. The manifest retains the private text key and
should not be published as a secret-key deployment configuration.

The focused watermarking suites passed 14 CPU checks in this checkout, including
hard-text surrogate gradients, frozen-head gradients, padding, KL normalization,
partition leakage and Qwen's hybrid cache. The replacement's real-Qwen GPU
preflight is queued and gates the remaining jobs. For the external teacher,
human holdout perplexity describes the underlying unhooked Qwen weights;
reference-generation perplexity and the teacher KL measure the marked outputs.
