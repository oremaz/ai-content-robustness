# RL Evasion & Arms Race: Literature-Driven Improvement Plan

Based on the latest proceedings from **ICML 2025**, **NeurIPS 2025**, and **ICLR 2026**, the adversarial evasion landscape has evolved from simple post-hoc modifications to formal game-theoretic frameworks and decoding-time interventions. Here is the detailed plan to modernize the `rl_evasion` framework.

## Phase 1: Game-Theoretic Equilibrium & Adaptation Tracking
**Context (ICLR 2026 - StealthRL; ICML 2025 - DeepScientist):** 
Recent literature treats the attacker-defender arms race as a formal two-player minimax game, proving convergence to a Nash equilibrium. It is no longer sufficient to merely show that an attacker can beat a detector; one must measure the *Adaptation Cost* (how many steps it takes to recover evasion after the defender updates) and the *Equilibrium Gap*.
**Implementation Details:**
- **Update `arms_race/equilibrium.py`:** Add explicit metrics for the "Equilibrium Gap" (Defender AUROC minus Attacker Evasion Success) at each round.
- **Track Adaptation Cost:** Measure how fast the attacker's evasion rate recovers during the fine-tuning phase of each round to plot an adaptation cost curve.

## Phase 2: Diffusion Purification for Image Evasion
**Context (ICML 2025 - Saberi et al.; NeurIPS 2025 - Unwinnable Arms Race):** 
The state-of-the-art for evading image detectors (especially frequency-based ones like WaRPAD) is "Diffusion Purification". This involves adding a small amount of Gaussian noise to the generated image and passing it through a few steps of a standard reverse-diffusion process to scrub adversarial artifacts and high-frequency signatures.
**Implementation Details:**
- **Update `image_evasion/ddpo_trainer.py`:** Introduce a `DiffusionPurifier` module. 
- During evaluation, run images through the purifier (via an Image2Image pipeline mechanism) and measure the delta in evasion score before and after purification against the `FrequencyDetector` and `PatchBasedClassifier`.

## Phase 3: Decoding-Time Intervention (HUMPA-style Proxy Evasion)
**Context (NeurIPS 2025 - HUMPA):**
Rather than full RL fine-tuning or post-hoc paraphrasing, the latest text evasion methods use an RL-trained Small Language Model (SLM) as a proxy. This proxy intervenes during the decoding of a larger target LLM, shifting the output logits at inference time to align with human token distributions.
**Implementation Details:**
- **Update `text_evasion/`:** Add a wrapper that implements proxy-guided decoding, adjusting the base model's next-token probabilities using the proxy model's logits scaled by a dynamic intervention factor.
