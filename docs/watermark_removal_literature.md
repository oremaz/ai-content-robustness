# Removing an unknown watermark: literature and experiment selection

Reviewed on 2026-10-26. The target is a model whose weights and ordinary
architecture are accessible, while its watermark technology, key, message and
extractor remain unknown. An API-only generator is a different access setting.

The shortlist contains **two model-level method families**. They require no
watermark detector queries. These are research candidates, not established
universal removers.

| Method | Clean weights available | Clean weights unavailable |
| --- | --- | --- |
| Merge toward the compatible clean checkpoint | Primary inexpensive attack; use a small fixed interpolation sweep | Ineligible; do not quietly substitute a related checkpoint |
| Continued training on independent clean data | Eligible without using the clean weights; permits comparison with merging | Primary attack, starting only from the watermarked weights |

For merging, full replacement is a sanity control. The meaningful question is
whether partial merging removes detectable signals while retaining useful
capabilities. A parameter difference between two checkpoints does not identify
which changes encode a watermark.

For continued training, use the architecture's ordinary objective: next-token
prediction for an LLM or autoregressive image backbone, and denoising for a
diffusion backbone. Broad representative data are the primary choice. Existing
LoRA runs provide a cheaper baseline; full-parameter training is the stronger
candidate where resources permit. These are variants within one family, not an
additional collection of attacks. The trainable modules must be chosen from the
ordinary training recipe, without consulting the actual watermark location.
Failure against a watermark in another component must remain a reported failure.

## Evidence from full papers

Full PDFs, including appendices, were downloaded and read. Local research assets
are under `results/research_workflow/watermarking/resources/`.

- **[Learning to Watermark in the Latent Space of Generative Models](https://proceedings.mlr.press/v306/rebuffi26a.html), ICML 2026.**
  The peer-reviewed DistSeal version directly studies clean-data LoRA forgetting
  in Appendix G. After 2,500 steps on 50 images of one class, the strongest
  reported learning rate reduces mean bit accuracy to roughly 0.70 for diffusion
  and 0.65 for RAR. This supports testing clean training on our image backbones.
  It establishes weakening, not complete removal: bit accuracy is not the
  detection rate at a fixed false-positive rate. The experiment also does not
  establish broad-domain removal or preserved quality after every attack.

- **[Making Open-Source Text LLM Watermarks Durable Against Merging](https://www.sri.inf.ethz.ch/publications/scharff26mergewm), NeurIPS 2026 listing.**
  The authors' public 26-page PDF reports substantial removal by merging with
  clean weights, including standard AAR/KTH distilled watermarks. Crucially,
  their merge-adversarial training preserves much more signal under the same
  operations. Merging is therefore a strong candidate with explicit
  counterexamples to universality. Appendix A.2 also shows ordinary fine-tuning
  substantially weakens detection, with domain-dependent effects. The title is
  listed in the conference's [2026 downloads](https://neurips.cc/Downloads/2026);
  the downloadable author PDF is still labeled "Preprint."

- **[Towards Watermarking of Open-Source LLMs](https://www.sri.inf.ethz.ch/publications/gloaguen2025oss), WMARK workshop at ICLR 2025.**
  This is workshop evidence, not an ICLR main-conference paper. Its systematic
  comparison supports merging and broad-domain clean training. Full training
  weakens the tested watermarks more consistently than LoRA; task-specific
  training can leave substantial signal outside the training domain. Ordinary
  quantization and pruning often preserve the watermark. This motivates
  excluding a broad compression/pruning grid from the initial shortlist.

- **[Black-Box Detection of Language Model Watermarks](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0f7ee1901fef1da4dae2b88dfd43195-Abstract-Conference.html), ICLR 2025.**
  Prompt-based statistical tests detect evidence of three watermark families
  without the private key, a clean counterpart or detector queries. They test
  context-dependent preferences, repeated-output diversity and cache effects.
  Their scope is those mechanisms and their modeling assumptions; they do not
  establish a detector for arbitrary unseen watermark technologies. Their
  instruction-following setup and family-specific assumptions need validation
  before applying them to Qwen Base, TextSeal or our learned latent adaptation.

- **[When There Is No Decoder](https://doi.org/10.1007/978-981-95-3537-8_30), ICICS 2025.**
  The full author version is arXiv:2507.03646. Despite its no-box setting, its
  fine-tuning attack trains a surrogate watermark extractor and an alternative
  payload. Results depend on assumptions about the extractor's capacity; some
  image-quality comparisons use only 20 images. It is less suitable as our
  first scheme-agnostic method than ordinary clean training, and is not selected.

## Detection, selection and verification

Knowing that a model is watermarked does not require first recovering its
technology before attempting removal. Without an owner detector, a negative
generic test cannot certify that an unknown watermark is absent. A classifier
that separates real from generated content is not a watermark detector.

With a clean counterpart, paired-output differences or a learned classifier can
measure distribution changes, but those changes can include ordinary fine-tuning
effects. Without the clean counterpart, that paired comparison is unavailable.
The existing base/student text classifier is therefore an evaluator diagnostic,
not an attacker capability in the no-clean-weights setup.

The implementation uses separate inputs and outputs:

1. The attacker receives the watermarked model, architecture/tokenizer, its
   independent clean fitting data, a fixed compute budget and utility checks.
   Only the first setup additionally receives clean model weights. It receives
   no watermark identifier, implementation, key, payload, extractor or score.
2. Attack strengths and checkpoints are fixed beforehand or selected using
   clean validation utility. Owner detection results cannot select the attack,
   its hyperparameters, updated modules or stopping point.
3. A separate evaluator applies the actual owner detector after artifacts are
   frozen, using held-out inputs and a frozen threshold. Report detection,
   false positives with uncertainty, quality and compute, including performance
   outside the fitting domain. Suppression of a measured signal is the supported
   claim; universal erasure is not.

The original merge and clean-training jobs are useful baselines. Their training
losses do not use watermark scores, but training and owner evaluation currently
share orchestration and some stages expose fixed intermediate measurements.
They should not be described as an enforced blind access boundary. The new
`scripts/watermark_removal.py` workflow reuses their training loops with separate
validated worker contracts and a later owner-evaluation process. It preserves
the original jobs and their frozen snapshots. The boundary controls inputs and
feedback in this research protocol; it is not a security sandbox for arbitrary
code running under the same filesystem account.

TextSeal is a separate sampler-level control: its Qwen weights are unchanged.
Weight merging alone does not remove a watermark reapplied by an external
sampler. Ordinary local inference tests sampler bypass; it is not evidence of
removing an unknown in-model watermark. Per-output rewriting or image
regeneration would likewise clean individual outputs rather than establish
model-weight removal, so they are outside this initial two-method shortlist.
