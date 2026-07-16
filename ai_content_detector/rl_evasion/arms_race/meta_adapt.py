"""Policy-gradient MAML for fast adversarial adaptation.

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

    def __init__(self, seed: int = 42):
        self.detectors: Dict[str, callable] = {}
        self._rng = np.random.default_rng(seed)

    def add(self, name: str, detector_fn: callable):
        """Add a detector. detector_fn: text -> float (AI probability)."""
        self.detectors[name] = detector_fn
        logger.info("Detector zoo: added %s (total: %d)", name, len(self.detectors))

    def sample(self, k: int = 3) -> Dict[str, callable]:
        """Sample k detectors from the zoo."""
        names = list(self.detectors.keys())
        if not names:
            raise ValueError("Cannot sample from an empty detector zoo")
        selected = self._rng.choice(names, size=min(k, len(names)), replace=False)
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
            zoo.add("roberta_openai", lambda t, d=dr: 1.0 - d(t))
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


def _functional_forward(
    model,
    input_ids,
    fast_weights: Dict[str, torch.Tensor],
    attention_mask=None,
    labels=None,
):
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
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if labels is not None:
            kwargs["labels"] = labels
        return functional_call(
            model, params_and_buffers,
            args=(),
            kwargs=kwargs,
        )

    # Fallback: in-place swap. OK for FOMAML (first-order) only.
    saved = {}
    for name, param in model.named_parameters():
        if name in fast_weights:
            saved[name] = param.data.clone()
            param.data = fast_weights[name].detach()
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
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
        max_new_tokens: int = 64,
        rollouts_per_prompt: int = 2,
        seed: int = 42,
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
        self.max_new_tokens = max_new_tokens
        self.rollouts_per_prompt = rollouts_per_prompt
        self.seed = seed
        self.adaptation_curves = []
        self._initial_adaptable_state = {
            name: param.detach().clone()
            for name, param in _get_adaptable_params(model)
        }

    def _sample_completion(
        self,
        model,
        prompt: str,
        fast_weights: Optional[Dict[str, torch.Tensor]],
        seed: int,
    ) -> Tuple[str, torch.Tensor]:
        """Sample an action and return its differentiable mean log-probability."""
        device = next(model.parameters()).device
        encoded = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256,
        ).to(device)
        prompt_len = encoded["input_ids"].shape[1]
        sequence = encoded["input_ids"]
        generator = torch.Generator(device=device).manual_seed(seed)
        eos_id = self.tokenizer.eos_token_id

        model.eval()
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                attention = torch.ones_like(sequence)
                if fast_weights is None:
                    outputs = model(input_ids=sequence, attention_mask=attention)
                else:
                    outputs = _functional_forward(
                        model, sequence, fast_weights, attention_mask=attention,
                    )
                logits = outputs.logits[:, -1] / 0.8
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1, generator=generator)
                sequence = torch.cat([sequence, next_token], dim=1)
                if eos_id is not None and int(next_token.item()) == eos_id:
                    break

        attention = torch.ones_like(sequence)
        if fast_weights is None:
            outputs = model(input_ids=sequence, attention_mask=attention)
        else:
            outputs = _functional_forward(
                model, sequence, fast_weights, attention_mask=attention,
            )
        token_log_probs = torch.log_softmax(outputs.logits[:, :-1], dim=-1)
        targets = sequence[:, 1:]
        selected = token_log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        completion_log_prob = selected[:, prompt_len - 1:].mean()
        text = self.tokenizer.decode(sequence[0, prompt_len:], skip_special_tokens=True)
        return text, completion_log_prob

    def _compute_evasion_loss(
        self,
        model,
        detector_fn: callable,
        prompts: List[str],
        fast_weights: Optional[Dict[str, torch.Tensor]] = None,
        num_samples: int = 4,
    ) -> torch.Tensor:
        """Return a REINFORCE loss over detector-scored generated completions."""
        if not prompts:
            raise ValueError("MAML evasion loss requires at least one prompt")
        losses = []
        sample_counter = 0
        for prompt_idx, prompt in enumerate(prompts[:num_samples]):
            rewards = []
            log_probs = []
            for rollout in range(self.rollouts_per_prompt):
                text, log_prob = self._sample_completion(
                    model,
                    prompt,
                    fast_weights,
                    seed=self.seed + 1009 * prompt_idx + rollout + sample_counter,
                )
                detector_score = float(detector_fn(text))
                if not np.isfinite(detector_score) or not 0.0 <= detector_score <= 1.0:
                    raise ValueError(f"Detector returned invalid AI score: {detector_score}")
                rewards.append(1.0 - detector_score)
                log_probs.append(log_prob)
                sample_counter += 1
            reward_tensor = torch.tensor(rewards, device=log_probs[0].device)
            baseline = reward_tensor.mean() if len(rewards) > 1 else reward_tensor.new_tensor(0.5)
            advantages = reward_tensor - baseline
            losses.extend(-adv.detach() * log_prob for adv, log_prob in zip(advantages, log_probs))
        return torch.stack(losses).mean()

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
        lora_params = _get_lora_params(model)

        # Initialize fast weights from current model parameters
        fast_weights = {name: param.clone() for name, param in lora_params}

        evasion_scores = []

        for step in range(k_steps):
            start = (step * 2) % len(prompts)
            step_prompts = [prompts[(start + offset) % len(prompts)] for offset in range(min(2, len(prompts)))]
            # Compute loss at current fast_weights
            loss = self._compute_evasion_loss(
                model, detector_fn,
                step_prompts,
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

            # Track evasion under the adapted weights.
            prompt = prompts[step % len(prompts)]
            text, _ = self._sample_completion(
                model, prompt, fast_weights, seed=self.seed + 50_000 + step,
            )
            score = float(detector_fn(text))
            evasion_scores.append(1.0 - score)

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

        Inner and outer losses use the same score-function policy-gradient
        objective on different prompt splits.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        adaptable = _get_adaptable_params(self.model)
        if not adaptable:
            raise RuntimeError(
                "meta_adapt: no adaptable parameters found. Apply LoRA or set "
                "requires_grad=True on the parameters you want to meta-learn."
            )
        if len(self.detector_zoo.detectors) == 0:
            raise ValueError("Meta-training requires a nonempty detector zoo")
        if len(prompts) < 4:
            raise ValueError("Meta-training requires at least four prompts")
        meta_optimizer = torch.optim.Adam(
            [param for _, param in adaptable],
            lr=self.meta_lr,
        )

        order_str = "FOMAML (first-order)" if self.first_order else "full second-order MAML"
        logger.info("Starting meta-training with %s, %d outer steps", order_str, self.outer_steps)

        meta_losses_history = []
        rng = np.random.default_rng(self.seed)

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
                [param for _, param in adaptable],
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
                for prompt_idx, p in enumerate(prompts[:20]):
                    text, _ = self._sample_completion(
                        self.model, p, None, seed=self.seed + 70_000 + prompt_idx,
                    )
                    score = float(held_out_detector(text))
                    evasion_scores.append(1.0 - score)

                mean_evasion = float(np.mean(evasion_scores))
                curve["steps"].append(step)
                curve["evasion"].append(mean_evasion)
                logger.info("  Adaptation step %d: evasion=%.3f", step, mean_evasion)

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

    def compare_with_pre_meta(
        self,
        held_out_detector: callable,
        detector_name: str,
        prompts: List[str],
        max_steps: int = 100,
    ) -> Dict:
        """Compare meta-learned adaptation with the exact pre-meta initialization.

        Returns both curves for plotting.
        """
        logger.info("Measuring meta-learned adaptation cost...")
        meta_curve = self.measure_adaptation_cost(
            held_out_detector, f"{detector_name}_meta", prompts, max_steps
        )

        # Compare against the exact pre-meta initialization under the same protocol.
        logger.info("Measuring pre-meta initialization adaptation cost...")
        saved_state = {
            name: param.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._initial_adaptable_state:
                    param.copy_(self._initial_adaptable_state[name])

        pre_meta_curve = self.measure_adaptation_cost(
            held_out_detector, f"{detector_name}_pre_meta", prompts, max_steps
        )

        # Restore
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in saved_state:
                    param.copy_(saved_state[name])

        return {
            "meta": meta_curve,
            "pre_meta": pre_meta_curve,
            "speedup": self._compute_speedup(meta_curve, pre_meta_curve),
        }

    # Compatibility alias for callers written before the baseline was corrected.
    compare_with_scratch = compare_with_pre_meta

    def _compute_speedup(self, meta_curve: Dict, pre_meta_curve: Dict) -> float:
        """Compute how much faster meta-learning adapts vs. its initialization.

        Defined as: steps_pre_meta_to_threshold / steps_meta_to_threshold
        where threshold = 0.7 evasion.
        """
        threshold = 0.7

        def steps_to_threshold(curve):
            for step, evasion in zip(curve["steps"], curve["evasion"]):
                if evasion >= threshold:
                    return step
            return curve["steps"][-1] if curve["steps"] else float("inf")

        meta_steps = steps_to_threshold(meta_curve)
        pre_meta_steps = steps_to_threshold(pre_meta_curve)

        if meta_steps == 0:
            return float("inf")
        return pre_meta_steps / meta_steps
