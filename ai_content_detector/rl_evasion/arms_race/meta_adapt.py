"""MAML-style meta-learning for fast adversarial adaptation.

Implements full second-order MAML (Finn et al., ICML 2017) over a "detector zoo"
so the attacker learns a meta-initialization that adapts in few gradient steps
to any new detector.

Bi-level objective:
    min_theta  sum_Ti  L_Ti(theta_i')
    where  theta_i' = theta - alpha * grad_theta L_Ti(theta)

Meta-gradient (second-order):
    grad_theta sum_Ti L_Ti(theta_i')
    = sum_Ti grad_{theta_i'} L_Ti(theta_i') * (I - alpha * H_Ti(theta))

The Hessian-vector product is computed implicitly via torch.autograd.grad
with create_graph=True, allowing differentiation through the inner loop.

FOMAML approximation (first-order): drops the Hessian term, sets
d(theta_i')/d(theta) = I.

References:
    - Finn et al., "Model-Agnostic Meta-Learning" (ICML 2017, arXiv:1703.03400)
    - Wang et al., "Fast Adversarial Robustness Adaptation" (ICLR 2021)
    - Rivera Soto et al., style embeddings for detector zoo diversity
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DetectorZoo:
    """A population of detectors covering multiple feature families.

    Used as the "task distribution" for meta-learning. Each detector is
    a task; the meta-learner must adapt to fool any detector in the zoo.
    """

    def __init__(self):
        self.detectors: Dict[str, callable] = {}

    def add(self, name: str, detector_fn: callable):
        """Add a detector. detector_fn: text -> float (AI probability)."""
        self.detectors[name] = detector_fn
        logger.info("Detector zoo: added %s (total: %d)", name, len(self.detectors))

    def sample(self, k: int = 3) -> Dict[str, callable]:
        """Sample k detectors from the zoo."""
        names = list(self.detectors.keys())
        selected = np.random.choice(names, size=min(k, len(names)), replace=False)
        return {n: self.detectors[n] for n in selected}

    def get_all(self) -> Dict[str, callable]:
        return dict(self.detectors)

    @classmethod
    def build_default(cls, device: str = "auto") -> "DetectorZoo":
        """Build a default detector zoo with available detectors."""
        zoo = cls()

        try:
            from ...detectors.text_detectors import BinocularsDetector
            det = BinocularsDetector(device=device)
            if det.is_available():
                zoo.add("binoculars", lambda t, d=det: d.detect(t).score)
        except Exception:
            pass

        try:
            from ...detectors.text_detectors import FastDetectGPTDetector
            det = FastDetectGPTDetector(device=device)
            if det.is_available():
                zoo.add("fast_detect_gpt", lambda t, d=det: d.detect(t).score)
        except Exception:
            pass

        try:
            from ...detectors.text_detectors import DivEyeDetector
            det = DivEyeDetector(device=device)
            if det.is_available():
                zoo.add("diveye", lambda t, d=det: d.detect(t).score)
        except Exception:
            pass

        try:
            from ..text_evasion.rewards import _build_roberta_detector_reward
            dr = _build_roberta_detector_reward(device)
            zoo.add("roberta_openai", lambda t, d=dr: d(t))
        except Exception:
            pass

        # Style embedding detector (Rivera Soto et al., 2024)
        try:
            from ...detectors.style_detector import StyleEmbeddingDetector
            det = StyleEmbeddingDetector(device=device)
            if det.is_available():
                zoo.add("style_embedding", lambda t, d=det: d.detect(t).score)
        except Exception:
            pass

        return zoo


# ---------------------------------------------------------------------------
# Functional forward pass utilities
# ---------------------------------------------------------------------------


def _get_adaptable_params(model) -> List[Tuple[str, torch.nn.Parameter]]:
    """Get named parameters the inner loop can adapt.

    Prefers LoRA (adapter) parameters when present, falls back to every
    trainable parameter so non-PEFT models don't silently produce a no-op
    inner loop.
    """
    lora = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and "lora" in name.lower()
    ]
    if lora:
        return lora
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


# Backwards-compat alias — existing tests/callers may still import this name.
_get_lora_params = _get_adaptable_params


def _functional_forward(model, input_ids, fast_weights: Dict[str, torch.Tensor]):
    """Forward pass using ``fast_weights`` in place of the model's parameters.

    Requires ``torch.func.functional_call`` so the fast-weight tensors remain in
    the autograd graph (a plain ``param.data = ...`` swap would detach them and
    silently degrade second-order MAML to a no-op). When ``torch.func`` isn't
    available we fall back to an in-place swap, which is correct only for
    FOMAML.
    """
    try:
        from torch.func import functional_call  # PyTorch >= 2.0
    except Exception:
        functional_call = None

    if functional_call is not None:
        # Build a complete parameter dict: fast weights where present, original
        # parameters (no .data copy — graph preserved) elsewhere.
        params_and_buffers = {}
        for name, param in model.named_parameters():
            params_and_buffers[name] = fast_weights.get(name, param)
        for name, buf in model.named_buffers():
            params_and_buffers[name] = buf
        return functional_call(
            model, params_and_buffers,
            args=(),
            kwargs={"input_ids": input_ids, "labels": input_ids},
        )

    # Fallback: in-place swap. OK for FOMAML (first-order) only.
    saved = {}
    for name, param in model.named_parameters():
        if name in fast_weights:
            saved[name] = param.data.clone()
            param.data = fast_weights[name].detach()
    try:
        outputs = model(input_ids=input_ids, labels=input_ids)
    finally:
        for name, param in model.named_parameters():
            if name in saved:
                param.data = saved[name]
    return outputs


class MAMLAdaptation:
    """Full second-order MAML for fast adaptation to new detectors.

    The meta-objective: learn a LoRA initialization such that K gradient
    steps of adaptation against any detector from the zoo achieves good
    evasion.

    Outer loop: meta-update on the LoRA initialization via meta-gradient
    Inner loop: K adaptation steps against a sampled detector subset

    The key difference from the previous implementation: the inner loop
    uses torch.autograd.grad(create_graph=True) so that the meta-gradient
    can flow through the inner loop updates (Hessian-vector products).

    With first_order=True, this reduces to FOMAML (drops the Hessian term).
    """

    def __init__(
        self,
        model,
        tokenizer,
        detector_zoo: DetectorZoo,
        meta_lr: float = 1e-4,
        inner_lr: float = 5e-5,
        inner_steps: int = 5,
        outer_steps: int = 100,
        detectors_per_episode: int = 3,
        first_order: bool = False,
        gradient_clip: float = 1.0,
        output_dir: str = "results/meta_adapt",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.detector_zoo = detector_zoo
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.detectors_per_episode = detectors_per_episode
        self.first_order = first_order
        self.gradient_clip = gradient_clip
        self.output_dir = output_dir
        self.adaptation_curves = []

    def _compute_evasion_loss(
        self,
        model,
        detector_fn: callable,
        prompts: List[str],
        fast_weights: Optional[Dict[str, torch.Tensor]] = None,
        num_samples: int = 4,
    ) -> torch.Tensor:
        """Compute differentiable evasion loss for a detector.

        Generate text from the model, score with the detector, and return
        a differentiable loss. The loss is the cross-entropy scaled by
        how detectable the generated text is.

        For the meta-gradient to flow, we use the language modeling loss
        (which IS differentiable w.r.t. model params) weighted by the
        detector score (which provides the reward signal).
        """
        device = next(model.parameters()).device

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for prompt in prompts[:num_samples]:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(device)

            # Generate text (non-differentiable — provides reward signal)
            model.eval()
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=200, do_sample=True, temperature=0.8,
                )
            text = self.tokenizer.decode(
                gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )

            # Get detector score (reward signal, non-differentiable)
            try:
                detector_score = detector_fn(text)
            except Exception:
                detector_score = 0.5

            # Compute differentiable language modeling loss
            model.train()
            if fast_weights is not None:
                outputs = _functional_forward(model, inputs["input_ids"], fast_weights)
            else:
                outputs = model(**inputs, labels=inputs["input_ids"])

            # Scale CE loss by detector score: higher detection -> higher loss
            # This creates the gradient signal: "change parameters to reduce detectability"
            total_loss = total_loss + outputs.loss * detector_score

        return total_loss / num_samples

    def _differentiable_inner_loop(
        self,
        model,
        detector_fn: callable,
        prompts: List[str],
        k_steps: int,
    ) -> Tuple[Dict[str, torch.Tensor], List[float]]:
        """Run K inner-loop adaptation steps with differentiable parameter updates.

        Uses torch.autograd.grad(create_graph=True) to maintain the computation
        graph through the inner loop, enabling second-order meta-gradients.

        Returns:
            (fast_weights, evasion_scores) — adapted parameters and per-step scores
        """
        device = next(model.parameters()).device
        lora_params = _get_lora_params(model)

        # Initialize fast weights from current model parameters
        fast_weights = {name: param.clone() for name, param in lora_params}

        evasion_scores = []

        for step in range(k_steps):
            # Compute loss at current fast_weights
            loss = self._compute_evasion_loss(
                model, detector_fn,
                prompts[step * 2:(step + 1) * 2],  # cycle through prompts
                fast_weights=fast_weights,
                num_samples=2,
            )

            # Compute gradients w.r.t. fast_weights
            # create_graph=True for second-order MAML (Hessian-vector products)
            # create_graph=False for FOMAML (first-order approximation)
            grad_list = torch.autograd.grad(
                loss,
                list(fast_weights.values()),
                create_graph=not self.first_order,
                allow_unused=True,
            )

            # Update fast weights: theta' = theta - alpha * grad
            new_fast_weights = {}
            for (name, _), grad in zip(lora_params, grad_list):
                if grad is not None:
                    new_fast_weights[name] = fast_weights[name] - self.inner_lr * grad
                else:
                    new_fast_weights[name] = fast_weights[name]
            fast_weights = new_fast_weights

            # Track evasion (non-differentiable, for logging)
            with torch.no_grad():
                prompt = prompts[step % len(prompts)]
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=256
                ).to(device)
                gen = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.8)
                text = self.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                try:
                    score = detector_fn(text)
                    evasion_scores.append(1.0 - score)
                except Exception:
                    evasion_scores.append(0.5)

        return fast_weights, evasion_scores

    def train(self, prompts: List[str]):
        """Run full second-order MAML meta-training.

        For each meta-step:
        1. Sample detectors from the zoo (tasks).
        2. Split prompts into disjoint support/query splits per episode.
        3. For each detector, run K differentiable inner-loop steps on SUPPORT
           to obtain theta'.
        4. Compute post-adaptation loss L(theta') on QUERY (a different sample),
           which is the correct MAML bi-level signal.
        5. Backprop through the full computation graph (including inner loop).

        With first_order=True (FOMAML), step 5 only uses first-order gradients.

        Note on the inner/outer loss: since text generation is non-differentiable,
        the "loss" used here is an evasion-weighted LM loss (CE·detector_score),
        a supervised surrogate for the true RL objective. Inner and outer loops
        use the *same* surrogate definition, only on different prompt splits.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        adaptable = _get_adaptable_params(self.model)
        if not adaptable:
            raise RuntimeError(
                "meta_adapt: no adaptable parameters found. Apply LoRA or set "
                "requires_grad=True on the parameters you want to meta-learn."
            )
        meta_optimizer = torch.optim.Adam(
            [param for _, param in adaptable],
            lr=self.meta_lr,
        )

        order_str = "FOMAML (first-order)" if self.first_order else "full second-order MAML"
        logger.info("Starting meta-training with %s, %d outer steps", order_str, self.outer_steps)

        meta_losses_history = []
        rng = np.random.default_rng(0)

        for outer_step in range(self.outer_steps):
            meta_optimizer.zero_grad()

            # Sample detector subset (task distribution)
            sampled_detectors = self.detector_zoo.sample(self.detectors_per_episode)

            # Disjoint support/query split per episode for proper MAML semantics.
            perm = rng.permutation(len(prompts))
            half = max(2, len(prompts) // 2)
            support_prompts = [prompts[i] for i in perm[:half]]
            query_prompts = [prompts[i] for i in perm[half:half * 2]] or support_prompts

            total_meta_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
            step_evasions = []

            for det_name, det_fn in sampled_detectors.items():
                # Inner loop on SUPPORT -> adapted fast_weights
                fast_weights, evasion_scores = self._differentiable_inner_loop(
                    self.model,
                    det_fn,
                    support_prompts,
                    k_steps=self.inner_steps,
                )

                # Post-adaptation loss on QUERY: how well do adapted weights generalize?
                post_adapt_loss = self._compute_evasion_loss(
                    self.model, det_fn, query_prompts, fast_weights=fast_weights, num_samples=4,
                )

                total_meta_loss = total_meta_loss + post_adapt_loss
                step_evasions.append(np.mean(evasion_scores[-max(1, len(evasion_scores) // 3):]))

            # Average over tasks
            meta_loss = total_meta_loss / len(sampled_detectors)

            # Backprop the meta-gradient through the full computation graph
            # For second-order MAML, this computes Hessian-vector products
            # through the inner loop updates
            meta_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                [param for _, param in lora_params],
                max_norm=self.gradient_clip,
            )

            meta_optimizer.step()

            meta_losses_history.append(meta_loss.item())

            if (outer_step + 1) % 10 == 0:
                avg_evasion = np.mean(step_evasions)
                recent_loss = np.mean(meta_losses_history[-10:])
                logger.info(
                    "Meta-step %d/%d | Meta-loss: %.4f (avg10: %.4f) | "
                    "Avg evasion after adapt: %.3f | Detectors: %s",
                    outer_step + 1, self.outer_steps, meta_loss.item(),
                    recent_loss, avg_evasion,
                    list(sampled_detectors.keys()),
                )

        # Save meta-learned initialization
        meta_dir = os.path.join(self.output_dir, "meta_init")
        self.model.save_pretrained(meta_dir)
        self.tokenizer.save_pretrained(meta_dir)
        logger.info(
            "Meta-learned initialization saved to %s (method: %s)",
            meta_dir, order_str,
        )

    def measure_adaptation_cost(
        self,
        held_out_detector: callable,
        detector_name: str,
        prompts: List[str],
        max_steps: int = 100,
        measure_interval: int = 5,
    ) -> Dict:
        """Measure how quickly the meta-learned model adapts to a new detector.

        Returns adaptation cost curve: evasion score at each measurement point.
        """
        device = next(self.model.parameters()).device

        # Save current state
        saved_state = {
            name: param.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        curve = {"steps": [], "evasion": [], "detector": detector_name}
        adaptable = [p for _, p in _get_adaptable_params(self.model)]
        optimizer = torch.optim.SGD(adaptable, lr=self.inner_lr)

        for step in range(max_steps):
            if step % measure_interval == 0:
                # Measure evasion
                self.model.eval()
                evasion_scores = []
                for p in prompts[:20]:
                    inputs = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(device)
                    with torch.no_grad():
                        gen = self.model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.8)
                    text = self.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    try:
                        score = held_out_detector(text)
                        evasion_scores.append(1.0 - score)
                    except Exception:
                        evasion_scores.append(0.5)

                mean_evasion = float(np.mean(evasion_scores))
                curve["steps"].append(step)
                curve["evasion"].append(mean_evasion)
                logger.info("  Adaptation step %d: evasion=%.3f", step, mean_evasion)

            # Adaptation step — use the SAME evasion-weighted LM loss as the
            # meta-training inner loop so adaptation cost measures the same
            # objective that was optimized during meta-training (apples to apples).
            self.model.train()
            step_prompts = [prompts[step % len(prompts)]]
            loss = self._compute_evasion_loss(
                self.model, held_out_detector, step_prompts, fast_weights=None, num_samples=1,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Restore original state
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in saved_state:
                    param.copy_(saved_state[name])

        self.adaptation_curves.append(curve)
        return curve

    def compare_with_scratch(
        self,
        held_out_detector: callable,
        detector_name: str,
        prompts: List[str],
        max_steps: int = 100,
    ) -> Dict:
        """Compare meta-learned adaptation vs. training from scratch.

        Returns both curves for plotting.
        """
        logger.info("Measuring meta-learned adaptation cost...")
        meta_curve = self.measure_adaptation_cost(
            held_out_detector, f"{detector_name}_meta", prompts, max_steps
        )

        # For "from scratch" comparison, reset LoRA weights to random
        logger.info("Measuring from-scratch adaptation cost...")
        saved_state = {
            name: param.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Reinit LoRA weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and "lora" in name.lower():
                    torch.nn.init.kaiming_uniform_(param)

        scratch_curve = self.measure_adaptation_cost(
            held_out_detector, f"{detector_name}_scratch", prompts, max_steps
        )

        # Restore
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in saved_state:
                    param.copy_(saved_state[name])

        return {
            "meta": meta_curve,
            "scratch": scratch_curve,
            "speedup": self._compute_speedup(meta_curve, scratch_curve),
        }

    def _compute_speedup(self, meta_curve: Dict, scratch_curve: Dict) -> float:
        """Compute how much faster meta-learning adapts vs. scratch.

        Defined as: steps_scratch_to_threshold / steps_meta_to_threshold
        where threshold = 0.7 evasion.
        """
        threshold = 0.7

        def steps_to_threshold(curve):
            for step, evasion in zip(curve["steps"], curve["evasion"]):
                if evasion >= threshold:
                    return step
            return curve["steps"][-1] if curve["steps"] else float("inf")

        meta_steps = steps_to_threshold(meta_curve)
        scratch_steps = steps_to_threshold(scratch_curve)

        if meta_steps == 0:
            return float("inf")
        return scratch_steps / meta_steps
