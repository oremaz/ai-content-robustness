import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, load_checkpoint
from ai_content_detector.rl_evasion.config import ImageEvasionConfig
from ai_content_detector.rl_evasion.image_evasion.ddpo_trainer import DDPOImageEvasionTrainer, reference_similarity_reward
from ai_content_detector.rl_evasion.image_evasion.research import select_reference_scale
from scripts import image_experiments


def test_reference_reward_is_bounded_and_has_a_defined_midpoint():
    kl = torch.tensor([0.0, 0.1, 1.0, 10.0, 1e12])
    values = reference_similarity_reward(kl, 1.0)
    assert values[0] == 1 and values[2] == 0.5
    assert torch.all(values[:-1] > values[1:])
    assert torch.all((values >= 0) & (values <= 1))
    assert float((0.1 * values).max()) <= 0.10000001


@pytest.mark.parametrize('scale', [0, -1, float('inf'), float('nan')])
def test_reference_reward_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match='scale'):
        reference_similarity_reward(torch.tensor([0.0]), scale)


@pytest.mark.parametrize('values', [[-1.0], [float('nan')], [float('inf')]])
def test_reference_reward_rejects_invalid_kl(values):
    with pytest.raises(ValueError, match='KL'):
        reference_similarity_reward(torch.tensor(values), 1.0)


def test_reference_calibration_uses_positive_probe_median_and_rejects_zero_drift():
    assert select_reference_scale([0, 0.1, 0.2, 0.9]) == pytest.approx(0.2)
    assert reference_similarity_reward(torch.tensor([0.2]), 0.2).item() == pytest.approx(0.5)
    with pytest.raises(ValueError, match='measurable'):
        select_reference_scale([0, 0])
    with pytest.raises(ValueError, match='finite'):
        select_reference_scale([float('nan')])


def test_positive_reward_cannot_be_combined_with_legacy_direct_penalty():
    with pytest.raises(ValueError, match='not both'):
        DDPOImageEvasionTrainer(ImageEvasionConfig(kl_coeff=0.1))


def test_reference_reward_is_combined_before_advantage_normalization_and_saved(tmp_path, monkeypatch):
    from ai_content_detector.tests.test_training_restart import make_ddpo
    from ai_content_detector.rl_evasion.image_evasion import ddpo_trainer as module
    trainer = make_ddpo(tmp_path, monkeypatch, interrupt_after=3)
    trainer.config.reward_reference_weight = 0.1
    original = module.pipeline_sample_with_logprob

    def sample(**kwargs):
        kwargs['reference_kl_collector'].append(torch.tensor([1.0, 3.0]))
        return original(**kwargs)

    monkeypatch.setattr(module, 'pipeline_sample_with_logprob', sample)
    with pytest.raises(CheckpointPause):
        trainer.train()
    state = load_checkpoint(tmp_path / 'restart.pt')
    trajectory = state['progress']['trajectory']
    expected = trajectory['latents'][-1].flatten().numpy() + np.array([0.05, 0.025])
    np.testing.assert_allclose(trajectory['rewards'], expected)
    np.testing.assert_allclose(trajectory['advantages'], (expected - expected.mean()) / (expected.std() + 1e-8), atol=1e-6)
    assert trajectory['reward_statistics']['reference']['mean'] == pytest.approx(0.375)
    assert state['config']['kl_coeff'] == 0


def test_replacement_preserves_existing_jobs_and_links_prerequisites(tmp_path, monkeypatch):
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir()
    new.mkdir()
    previous = {'directory': str(old), 'jobs': {stage: {'job_id': str(i + 1)} for i, stage in enumerate(image_experiments.STAGES)}}
    previous_path = old / 'manifest.json'
    previous_path.write_text(json.dumps(previous))
    path = new / 'manifest.json'
    path.write_text(json.dumps({'directory': str(new), 'jobs': {}, 'protocol': {}}))
    monkeypatch.setattr(image_experiments, 'create_manifest', lambda: path)
    image_experiments.replacement_manifest(previous_path)
    result = json.loads(path.read_text())
    assert set(result['jobs']) == {'prepare', 'benchmark', 'warmup'}
    assert result['submission_stages'] == ['attack', 'defend']
    assert (new / 'partitions.json').is_symlink()
    assert (new / 'warmup').resolve() == old / 'warmup'
    assert result['jobs']['prepare']['job_id'] == '1'
    assert json.loads(previous_path.read_text()) == previous
    result['jobs']['attack'] = {'job_id': 'new_attack'}
    command = image_experiments.submission_command(path, result, 'defend')
    assert '--dependency=afterok:new_attack:2' in command


def test_submission_failure_never_cancels_adopted_jobs(tmp_path, monkeypatch):
    manifest = {'directory': str(tmp_path), 'jobs': {'prepare': {'job_id': '1'}, 'benchmark': {'job_id': '2'}, 'warmup': {'job_id': '3'}},
                'submission_stages': ['attack', 'defend']}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    responses = iter(['100\n', RuntimeError('submission failure')])
    def submit(*args, **kwargs):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value
    calls = []
    monkeypatch.setattr(image_experiments.subprocess, 'check_output', submit)
    monkeypatch.setattr(image_experiments.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    with pytest.raises(RuntimeError, match='submission failure'):
        image_experiments.submit(path, release=False)
    assert calls == [['scancel', '100']]


def test_sampler_collects_per_image_kl_only_on_stochastic_steps():
    from types import SimpleNamespace
    from diffusers import DDIMScheduler
    from ai_content_detector.tests.test_published_image import Policy
    from ai_content_detector.rl_evasion.image_evasion.ddpo_trainer import pipeline_sample_with_logprob
    policy = Policy()
    policy.device = torch.device('cpu')
    policy.config = SimpleNamespace(in_channels=1)
    pipeline = SimpleNamespace(unet=policy, scheduler=DDIMScheduler(clip_sample=False))
    collected = []
    with torch.no_grad():
        result = pipeline_sample_with_logprob(pipeline, torch.zeros(2, 1, 1), torch.zeros(2, 1, 1),
                                             num_inference_steps=4, reference_kl_collector=collected)
    assert len(result[1]) == 5
    assert len(collected) == 3
    assert all(values.shape == (2,) and torch.all(values > 0) for values in collected)
    assert policy.enabled
