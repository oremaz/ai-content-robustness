# Evasion and defender adaptation

This package studies a generator that receives detector feedback, and a detector
that may later retrain on the generated outputs. Separate modules cover text
optimization, diffusion-image optimization, adaptation to new detectors, and
benchmarking. The [retained results](../../docs/research_results.md) describe the
completed experiments; method availability alone is not an experimental result.

## Choose a path

| Question | Entry point | Main distinction |
|---|---|---|
| Train a text attacker | [text_evasion/grpo_trainer.py](text_evasion/grpo_trainer.py) | Optimizes sampled completions using detector and content rewards |
| Inspect the reward | [text_evasion/rewards.py](text_evasion/rewards.py) | Keeps detector evasion separate from content constraints |
| Study self-play-style text updates | [text_evasion/multispin.py](text_evasion/multispin.py) | Separate SPIN training and feature monitoring |
| Optimize an image generator | [image_evasion/ddpo_trainer.py](image_evasion/ddpo_trainer.py) | Diffusion trajectories, log probabilities and policy updates |
| Adapt to a new detector | [arms_race/meta_adapt.py](arms_race/meta_adapt.py) | Support updates and query loss have different training roles |
| Measure one defender update | [defender_comparison/workflow.py](defender_comparison/workflow.py) | Reuses identical saved evaluation texts before and after retraining |
| Compare methods and detectors | [benchmarking/benchmark.py](benchmarking/benchmark.py) | Validates dataset/method compatibility and held-out detector declarations |

Start with [config.py](config.py), the reward functions and the dedicated
workflow for the experiment of interest. The training entry points now delegate
to these smaller implementation modules:

| Module | Responsibility |
|---|---|
| [grpo_optimization.py](text_evasion/grpo_optimization.py) | TRL optimization and the separately named REINFORCE fallback |
| [diffusion_policy.py](image_evasion/diffusion_policy.py) | DDIM transition densities, reference KL and sampled trajectories |
| [image_rewards.py](image_evasion/image_rewards.py) | Detector/content rewards and prompt-local statistics |
| [ddpo_lifecycle.py](image_evasion/ddpo_lifecycle.py) | Model loading, prompt partitions and resumable evaluation |
| [ddpo_trainer.py](image_evasion/ddpo_trainer.py) | DDPO optimization and checkpoint progress |

Public trainer imports and reward/statistics class paths are preserved. The
common persistence primitives remain in [checkpointing.py](checkpointing.py).

## Evaluation boundaries

Split prompts before training. A held-out prompt is excluded from attacker and
defender fitting. In MAML, query prompts contribute to the meta-objective and
are therefore training data. A held-out detector is a separate kind of holdout;
checkpoint metadata records optimized detector names, but not all training sample
IDs. Declared detector overlap can be rejected; full sample-level overlap cannot
yet be rejected automatically.

The completed retained-attacker experiment contains one defender update for each
fixed attacker. Re-scoring the same saved texts isolates that update. It does not
establish resilience to another adaptive attack or a game-theoretic equilibrium.
Follow the [exact protocol](../../docs/defender_comparison.md) for data pairs,
checkpoint identities and continuation behavior.

## Run

From the repository root:

```bash
PYTHONPATH=. uv run python -m ai_content_detector.rl_evasion.run_experiments \
  --experiment smoke_test
PYTHONPATH=. uv run python -m pytest ai_content_detector/tests/test_config.py \
  ai_content_detector/tests/test_rewards.py ai_content_detector/tests/test_defender_comparison.py -q
```

The smoke test checks imports and wiring. Actual training needs the recorded
models, data, detectors and GPU environment. See the
[experiment command table](../../docs/research_guide.md#running-experiments)
for method-specific options. Resume an existing job with its original manifest,
frozen source and checkpoint state, rather than creating a new experiment from
today's defaults. Watermarking is a [separate package](../watermarking/README.md).
