"""HUMPA-style Decoding-Time Proxy Evasion (ICLR 2025).

Intervenes during the decoding of a target LLM, shifting next-token 
logits using an RL-trained proxy Small Language Model (SLM) to 
align output with human token distributions and evade detection.
"""
import torch
import torch.nn.functional as F

class ProxyEvasionWrapper:
    """Wraps a target LLM to inject proxy SLM logits during generation.
    
    This implements decoding-time intervention as described in HUMPA (Wang et al., ICLR 2025),
    modifying token probabilities without requiring fine-tuning of the base model.
    """
    
    def __init__(
        self,
        target_model,
        proxy_model,
        reference_proxy_model,
        intervention_alpha: float = 0.2,
    ):
        self.target_model = target_model
        self.proxy_model = proxy_model
        self.reference_proxy_model = reference_proxy_model
        self.intervention_alpha = intervention_alpha
        vocab_sizes = {
            int(model.config.vocab_size)
            for model in (target_model, proxy_model, reference_proxy_model)
        }
        if len(vocab_sizes) != 1:
            raise ValueError("HUMPA target, proxy, and reference proxy must share a vocabulary")
        self.proxy_model.eval()
        self.reference_proxy_model.eval()
        
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, **kwargs) -> torch.Tensor:
        """Custom generation loop with proxy intervention."""
        device = input_ids.device
        current_ids = input_ids.clone()
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)
        eos_id = self.target_model.config.eos_token_id
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Get target logits
                target_outputs = self.target_model(current_ids)
                target_logits = target_outputs.logits[:, -1, :]
                
                # Get proxy logits
                proxy_outputs = self.proxy_model(current_ids)
                proxy_logits = proxy_outputs.logits[:, -1, :]

                reference_outputs = self.reference_proxy_model(current_ids)
                reference_logits = reference_outputs.logits[:, -1, :]
                
                # Combine logits (HUMPA intervention)
                # alpha controls how much the proxy steers the target
                combined_logits = target_logits + self.intervention_alpha * (
                    proxy_logits - reference_logits
                )
                
                # Sample next token
                probs = F.softmax(combined_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                if eos_id is not None:
                    next_token = torch.where(
                        finished.unsqueeze(1),
                        torch.full_like(next_token, eos_id),
                        next_token,
                    )
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                if eos_id is not None:
                    finished |= next_token.squeeze(1).eq(eos_id)
                if bool(finished.all()):
                    break
                    
        return current_ids
