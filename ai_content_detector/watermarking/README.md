# Watermarking and removal

This package separates watermark creation, removal attempts and owner-side
evaluation. It covers the official TextSeal sampler, released DistSeal image
models and an experimental latent-text adaptation. These are different
watermark locations and threat models; see the [protocol](../../docs/watermarking.md)
for their exact relationship and current experiment status.

## Read the implementation

| File | Responsibility |
|---|---|
| [protocol.py](protocol.py) | Stage definitions, partitions, thresholds and metric contracts |
| [text.py](text.py) | TextSeal integration and text evaluation/distillation helpers |
| [latent_text.py](latent_text.py) | Experimental embedder/extractor and discrete text channel |
| [image.py](image.py) | Image generation and watermark evaluation |
| [backbones.py](backbones.py) | Native released model backbones and checkpoint state |
| [removal.py](removal.py) | Attacker tasks, artifact integrity and owner evaluation gating |
| [preflight.py](preflight.py) | Resource and environment checks before experiments |

Start with the protocol and removal contracts before following an individual
training worker. [watermark_experiments.py](../../scripts/watermark_experiments.py)
and [watermark_removal.py](../../scripts/watermark_removal.py) prepare manifests
and frozen source snapshots; their commands are in the protocol above.

## What the comparison means

Ordinary sampling can bypass a sampler-only watermark without changing model
weights. A watermark distilled into weights requires a different removal test.
Tasks with original clean weights can interpolate with that reference; tasks
without those weights use clean-data updates on the watermarked model. Owner
keys, watermark detector outputs and clean reference weights are excluded from
those no-reference worker contracts.

Those contracts separate process arguments and hash fixed artifacts; they are
not an operating-system sandbox against a malicious process on the same shared
account. The owner evaluates only after attack artifacts have been frozen.
Validation and final evaluation remain separate from clean-data fitting.

## Checks and reproduction

From the repository root:

```bash
PYTHONPATH=. uv run python -m pytest ai_content_detector/tests/test_watermarking.py \
  ai_content_detector/tests/test_watermark_removal.py ai_content_detector/tests/test_latent_watermarking.py -q
```

Some checks require the already prepared official source releases. These tests
do not establish successful watermark transfer or removal in a full model.
Pinned optional dependencies live in `requirements_watermarking.in`; keep the
recorded Torch/CUDA stack for an existing run. Resuming a pending experiment
uses its original frozen source and manifest. Do not replace them when updating
this documentation or another model family.
