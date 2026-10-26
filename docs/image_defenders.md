# Image defender candidates

Reviewed on 2026-10-25. The full PDFs, including technical appendices, were
downloaded and read for DDA (arXiv v6), Community Forensics (v2), and Effort (v4).
The results below are author-reported measurements, not repository results or
directly comparable rankings. DDA and Community Forensics wrappers are now
integrated. Community Forensics remains unused; the authorized official-DDA
benchmark and single-round SD1.5 experiment have a dedicated driver and
[live report](image_results.md). The previous third-party SigLIP integration
has been removed. DDA's official state dict passed a strict CPU load of all
537 keys; candidate benchmark results should be read from the live report.

## DDA: first candidate for the reward detector

[Paper, NeurIPS 2025](https://arxiv.org/pdf/2505.14359v6),
[official code, Apache-2.0](https://github.com/roy-ch/Dual-Data-Alignment),
[official weights](https://huggingface.co/Junwei-Xi/Dual-Data-Alignment).

DDA trains DINOv2 ViT-L/14 with LoRA on real MSCOCO images paired with
reconstructions from the Stable Diffusion 2.1 VAE. Matching JPEG compression
and mixing paired pixels reduce shortcuts based on encoding and content.
Inference is a classifier forward pass; reconstruction is for training data.
The paper reports balanced accuracy of 82.4% on Chameleon, 90.9% on SynthWildX,
and 90.3% on WildRF, alongside evaluations on eight other benchmarks.

The release corrected an earlier checkpoint in February 2026. Pin the new
checkpoint revision and hash. Its inference uses a 336-pixel center crop and
CLIP normalization, despite the DINOv2 backbone. Do not reuse our 200-pixel
ArtiFact preprocessing. The paper's Appendix A reports monitoring all evaluation
datasets during training with early stopping, so its benchmarks should not be
treated as a sealed selection protocol. Heavy postprocessing and smartphone
enhancement remain limitations. These results justify independent evaluation,
not an assumption of resistance to an adaptive attacker.

## Community Forensics: candidate for independent transfer evaluation

[Paper, CVPR 2025](https://arxiv.org/pdf/2411.04125v2),
[official code, MIT](https://github.com/JeongsooP/Community-Forensics),
[224-pixel weights](https://huggingface.co/OwensLab/commfor-model-224),
[384-pixel weights](https://huggingface.co/OwensLab/commfor-model-384).

This work addresses training diversity: its dataset contains 2.7 million
generated images from 4,803 generators, paired with real images. It trains a
ViT-S classifier end to end using diverse augmentations. Its own evaluation
holds out 21 generators. Table 2 reports 98.7% mean AP and 89.2% mean accuracy
on its comprehensive evaluation for the 384-pixel model. These metrics average
over generators; mean AP is not accuracy.

The released models have about 22 million parameters, making them attractive
for repeated scoring; actual throughput has not been measured here. Use the
authors' processor at the matching resolution and apply sigmoid to the binary
logit, with fake oriented upward. Their training set is dominated by related
latent diffusion models. Detection of base SD1.5 should not be described as
unseen-family generalization. This is a complementary training recipe and
backbone to DDA, without a guarantee of independent errors.

## Effort: additional research baseline

[Paper, ICML 2025](https://arxiv.org/pdf/2411.15633v4),
[official code and checkpoint links](https://github.com/YZY-stack/Effort-AIGI-Detection).

Effort preserves dominant pretrained CLIP weight components and learns in a
remaining subspace using orthogonality constraints. The ProGAN-trained paper
model reports 95.19% mean accuracy and 99.41% mean AP over 19 UniversalFakeDetect
subsets. A separate SD1.4-trained model reports 91.1% mean accuracy on GenImage,
including its seen SD1.4 subset. Those are different training protocols.

The release provides CLIP-L/14 general-image checkpoints through Google Drive.
Select the general-image checkpoint, rather than the face-forgery checkpoint,
and reproduce its preprocessing. The reviewed repository has no top-level
license declaration, so public availability alone does not establish the same
reuse terms as the two candidates above. Its older benchmark results also do
not establish performance on current generators or DDPO outputs.

## Proposed experiment

Evaluate DDA and Community Forensics separately before selecting a reward
ensemble. My recommendation is to try DDA as the fixed DDPO reward detector
and reserve Community Forensics for transfer scoring. This is an experimental
choice to validate, not a measured ranking.

1. Record exact weights, revisions, preprocessing, class direction, and seeds.
2. Score saved base-SD1.5 outputs and real images before any attacker update.
   Include external generators and matched resizing/compression checks.
3. Report AUROC, human false-positive rate, and TPR at fixed FPR targets.
   Choose thresholds on separate calibration data, then freeze them.
4. Keep prompt partitions and final evaluation images separate from training
   and detector selection. Do not use the transfer detector for reward or
   checkpoint selection if claiming held-out-detector transfer.
5. Run DDPO only after establishing a useful initial detector operating point.
   Report attack improvement against that baseline alongside prompt alignment.

The planned independent Community Forensics transfer evaluation is deferred.
The current authorized experiment uses DDA only. Performance claims for the new
local experiment must come from completed stage receipts and saved measurements.
