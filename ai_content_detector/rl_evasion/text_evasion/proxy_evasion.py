"""HUMPA-style Decoding-Time Proxy Evasion (NeurIPS 2025).

Intervenes during the decoding of a target LLM, shifting next-token 
logits using an RL-trained proxy Small Language Model (SLM) to 
align output with human token distributions and evade detection.
"""
import torch
import torch.nn.functional as F

class ProxyEvasionWrapper:
    """Wraps a target LLM to inject proxy SLM logits during generation.
    
    This implements decoding-time intervention as described in HUMPA (Wang et al., NeurIPS 2025),
    modifying token probabilities without requiring fine-tuning of the base model.
    """
    
    def __init__(self, target_model, proxy_model, intervention_alpha: float = 0.2):
        self.target_model = target_model
        self.proxy_model = proxy_model
        self.intervention_alpha = intervention_alpha
        
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, **kwargs) -> torch.Tensor:
        """Custom generation loop with proxy intervention."""
        device = input_ids.device
        current_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Get target logits
                target_outputs = self.target_model(current_ids)
                target_logits = target_outputs.logits[:, -1, :]
                
                # Get proxy logits
                proxy_outputs = self.proxy_model(current_ids)
                proxy_logits = proxy_outputs.logits[:, -1, :]
                
                # Combine logits (HUMPA intervention)
                # alpha controls how much the proxy steers the target
                combined_logits = target_logits + self.intervention_alpha * proxy_logits
                
                # Sample next token
                probs = F.softmax(combined_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                if next_token.item() == self.target_model.config.eos_token_id:
                    break
                    
        return current_ids
