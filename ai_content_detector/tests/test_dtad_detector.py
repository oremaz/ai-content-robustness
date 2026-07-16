from __future__ import annotations

import pytest
import torch

from ai_content_detector.detectors.image_detectors import DenoisingTrajectoryDetector


def test_ddim_inversion_update_reconstructs_clean_and_next_latent():
    clean = torch.tensor([[[[2.0]]]])
    noise = torch.tensor([[[[0.5]]]])
    alpha_current = torch.tensor(0.81)
    alpha_next = torch.tensor(0.49)
    current = alpha_current.sqrt() * clean + (1.0 - alpha_current).sqrt() * noise
    next_latent, predicted_clean = DenoisingTrajectoryDetector._ddim_inversion_update(
        current, noise, alpha_current, alpha_next,
    )
    expected_next = alpha_next.sqrt() * clean + (1.0 - alpha_next).sqrt() * noise
    assert predicted_clean.item() == pytest.approx(clean.item())
    assert next_latent.item() == pytest.approx(expected_next.item())


def test_dtad_defaults_to_paper_clip_intermediate_layer():
    detector = DenoisingTrajectoryDetector()
    assert detector._clip_model_id == "openai/clip-vit-large-patch14"
    assert detector._clip_layer == 15
    assert detector._trajectory_indices is None
