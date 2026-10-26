from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, load_checkpoint
from ai_content_detector.rl_evasion.config import ImageEvasionConfig
from ai_content_detector.rl_evasion.image_evasion.ablation import (
    choose_pilot, pilot_variants, scaled_logit_values, split_rows,
)
from ai_content_detector.rl_evasion.image_evasion.ddpo_trainer import grouped_advantages, rollout_prompt_indices
from scripts import image_attacker_ablation


def test_groups_remove_prompt_difficulty_and_constant_reference_offsets():
    prompts = ['hard', 'easy', 'hard', 'easy']
    a = grouped_advantages(prompts, [0.01, 0.9, 0.03, 0.8])
    np.testing.assert_allclose(a, [-1, 1, 1, -1], atol=1e-5)
    b = grouped_advantages(prompts, [0.01 + 0.04, 0.9 + 0.1, 0.03 + 0.04, 0.8 + 0.1])
    np.testing.assert_allclose(a, b, atol=1e-5)
    np.testing.assert_array_equal(grouped_advantages(['a', 'a'], [1, 1]), [0, 0])
    with pytest.raises(ValueError, match='two images'):
        grouped_advantages(['a', 'b'], [0, 1])


def test_rollout_groups_have_expected_coverage_and_reproducibility():
    indices = rollout_prompt_indices(800, 32, 4, 42)
    assert len(indices) == 32 and len(set(indices)) == 8
    assert np.all(indices.reshape(-1, 4) == indices.reshape(-1, 4)[:, :1])
    np.testing.assert_array_equal(indices, rollout_prompt_indices(800, 32, 4, 42))
    assert not np.array_equal(indices, rollout_prompt_indices(800, 32, 4, 43))
    with pytest.raises(ValueError):
        rollout_prompt_indices(800, 31, 4, 42)


def test_training_logit_scaling_preserves_differences_when_probabilities_saturate():
    logits = np.array([20, 25, 30], dtype=np.float32)
    assert len(set(torch.from_numpy(logits).sigmoid().tolist())) == 1
    values = scaled_logit_values(logits, center=-25, scale=5)
    assert 1 > values[0] > values[1] > values[2] > 0
    assert values[1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        scaled_logit_values(logits, -25, 0)


def test_positive_advantages_increase_expected_gaussian_action():
    from diffusers import DDIMScheduler
    from ai_content_detector.tests.test_published_image import Policy
    from ai_content_detector.rl_evasion.image_evasion.ddpo_trainer import ddim_step_with_logprob
    scheduler = DDIMScheduler(clip_sample=False)
    scheduler.set_timesteps(50)
    policy = Policy()
    latents, embeds = torch.ones(4096, 1, 1, 1), torch.zeros(8192, 1, 1)
    timestep = int(scheduler.timesteps[10])
    with torch.no_grad():
        actions, old, before, _ = ddim_step_with_logprob(scheduler, policy, latents, timestep, embeds,
                                                        generator=torch.Generator().manual_seed(42), return_stats=True)
    _, current = ddim_step_with_logprob(scheduler, policy, latents, timestep, embeds, prev_sample=actions)
    advantages = actions.flatten() - actions.mean()
    loss = -(advantages * torch.exp(current - old)).mean()
    optimizer = torch.optim.SGD(policy.parameters(), lr=1)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        _, _, after, _ = ddim_step_with_logprob(scheduler, policy, latents, timestep, embeds, prev_sample=actions, return_stats=True)
    assert after.mean() > before.mean()


def test_splits_reject_effective_prompt_overlap():
    data = {'fitting': [{'sample_id': str(i)} for i in range(1000)],
            'evaluation': [{'sample_id': str(1000 + i)} for i in range(500)]}
    assert list(map(len, split_rows(data))) == [800, 200, 500]
    data['evaluation'][0] = data['fitting'][0]
    with pytest.raises(ValueError, match='overlap'):
        split_rows(data)


def test_pilots_are_matched_and_reference_comparisons_change_only_reference():
    variants = pilot_variants()
    for variant in variants.values():
        config = asdict(ImageEvasionConfig())
        config.update(variant)
        assert config['num_train_epochs'] * config['sample_batch_size'] * config['rollout_batches'] == 800
    reference = []
    for name in ['combined_ref_000', 'combined_ref_003', 'combined_ref_010']:
        variant = dict(variants[name])
        reference.append(variant.pop('reward_reference_weight'))
        if name == 'combined_ref_000':
            expected = variant
        assert variant == expected
    assert reference == [0, 0.03, 0.1]


def test_pilot_selection_never_reads_test_data(tmp_path):
    for name, rate, mean in [('a', 0.1, 0.2), ('b', 0.2, 0.3)]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'selected.json').write_text(json.dumps({'evasion_rate': rate, 'mean_evasion': mean, 'clip_mean': 0.3}))
    assert choose_pilot({'directory': str(tmp_path), 'variants': {'a': {}, 'b': {}}}) == 'b'
    assert not (tmp_path / 'test').exists()


def test_chunked_rollout_restart_preserves_updates_and_diagnostics(tmp_path, monkeypatch):
    from ai_content_detector.tests.test_training_restart import make_ddpo
    def make(path, interrupt=None):
        trainer = make_ddpo(path, monkeypatch, interrupt_after=interrupt)
        trainer.config.rollout_batches = 2
        trainer.config.diagnostic_logging = True
        trainer.prompts = ['one', 'two', 'three', 'four', 'five']
        return trainer
    continuous = make(tmp_path / 'continuous')
    continuous.train()
    interrupted = make(tmp_path / 'restart', 9)
    with pytest.raises(CheckpointPause):
        interrupted.train()
    state = load_checkpoint(tmp_path / 'restart/restart.pt')
    assert len(state['progress']['trajectory']['rewards']) == 4
    resumed = make(tmp_path / 'restart')
    resumed.train()
    torch.testing.assert_close(resumed.pipeline.unet.weight, continuous.pipeline.unet.weight, rtol=0, atol=0)
    assert resumed.training_history == continuous.training_history
    assert resumed.training_history[-1]['sampled_images'] == 4
    assert 'clip_fraction' in resumed.training_history[-1]['optimization']


def test_slurm_graph_gates_all_pilots_then_long_then_test(tmp_path):
    manifest = {'directory': str(tmp_path), 'variants': pilot_variants(), 'jobs': {}}
    graph = image_attacker_ablation.stages(manifest)
    for i, stage in enumerate(graph):
        manifest['jobs'][stage] = {'job_id': str(100 + i)}
    assert graph['test'] == ['long']
    assert len(graph['long']) == 8
    for stage in graph['long']:
        assert graph[stage] == ['diagnostic']
    command = image_attacker_ablation.submission_command(tmp_path / 'manifest.json', manifest, 'long')
    assert '--time=12:00:00' in command and '--qos=qos_gpu_h100-t3' in command
    assert '--hold' in command
    assert '--dependency=afterok:' + ':'.join(manifest['jobs'][name]['job_id'] for name in graph['long']) in command


def test_failed_sweep_submission_cancels_only_new_jobs(tmp_path, monkeypatch):
    manifest = {'directory': str(tmp_path), 'variants': {'a': {}}, 'jobs': {},
                'parent_manifest': '/old/manifest.json'}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    calls = []
    answers = iter(['200', RuntimeError('submission failure')])
    def check(*args, **kwargs):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer
    monkeypatch.setattr(image_attacker_ablation.subprocess, 'check_output', check)
    monkeypatch.setattr(image_attacker_ablation.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    with pytest.raises(RuntimeError, match='submission failure'):
        image_attacker_ablation.submit(path)
    assert calls == [['scancel', '200']]


def test_frozen_worker_writes_report_to_original_repository(tmp_path, monkeypatch):
    (tmp_path / 'docs').mkdir()
    root = tmp_path / 'results/run'
    root.mkdir(parents=True)
    path = root / 'manifest.json'
    path.write_text(json.dumps({'repository': str(tmp_path), 'directory': str(root), 'variants': {}, 'jobs': {}}))
    monkeypatch.setattr(image_attacker_ablation, 'ROOT', root / 'source')
    image_attacker_ablation.report(path)
    assert (tmp_path / 'docs/image_attacker_ablation_results.md').exists()
    assert (root / 'report.md').exists()


@pytest.mark.parametrize('report_failure', [False, True])
def test_diagnostic_worker_keeps_metrics_separate_from_receipt(tmp_path, monkeypatch, report_failure):
    from types import SimpleNamespace
    from PIL import Image
    from diffusers import DDIMScheduler
    from ai_content_detector.tests.test_published_image import Policy
    from ai_content_detector.rl_evasion.image_evasion import ablation, research
    root = tmp_path / 'run'
    root.mkdir()
    (tmp_path / 'docs').mkdir()
    parent = tmp_path / 'parent'
    (parent / 'base_fitting').mkdir(parents=True)
    (parent / 'reference_reward_calibration.json').write_text('{}')
    (root / 'warmup').mkdir()
    checkpoint = root / 'warmup/completed.pt'
    checkpoint.write_bytes(b'checkpoint identity')
    image_path = parent / 'training.png'
    Image.new('RGB', (4, 4)).save(image_path)
    data = {'fitting': [{'sample_id': str(i), 'prompt': str(i)} for i in range(1000)],
            'evaluation': [{'sample_id': str(1000 + i), 'prompt': str(i)} for i in range(500)]}
    partitions = root / 'partitions.json'
    research.write_json(partitions, data)
    research.write_json(parent / 'base_fitting/images.json', {'rows': [
        {'sample_id': str(i), 'path': str(image_path), 'sha256': research.file_hash(image_path)} for i in range(256)]})
    policy = Policy()
    policy.device, policy.config = torch.device('cpu'), SimpleNamespace(in_channels=1)
    detector = SimpleNamespace(predict_logits=lambda images: np.arange(len(images), dtype=float))
    trainer = SimpleNamespace(pipeline=SimpleNamespace(unet=policy, scheduler=DDIMScheduler(clip_sample=False)),
                              reward_fn=SimpleNamespace(detector_rewards=[SimpleNamespace(detector=detector)]),
                              _encode_prompts=lambda prompts: (torch.zeros(len(prompts), 1, 1), torch.zeros(len(prompts), 1, 1)))
    monkeypatch.setattr(ablation, 'trainer_for', lambda *args: (trainer, data, root / 'control'))
    monkeypatch.setattr(ablation, 'evaluate', lambda *args: {'clip_mean': 0.3})
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    manifest = {'directory': str(root), 'repository': str(tmp_path), 'parent_directory': str(parent),
                'partitions': str(partitions), 'partitions_sha256': research.file_hash(partitions),
                'warmup_sha256': research.file_hash(checkpoint), 'code_sha256': {}, 'variants': {},
                'reference_calibration_sha256': research.file_hash(parent / 'reference_reward_calibration.json'),
                'resources': {'checkpoint': {'files': {'DDA_ckpt.pth': {'path': str(checkpoint), 'sha256': research.file_hash(checkpoint)}}}},
                'jobs': {'diagnostic': {'job_id': '123'}}}
    path = root / 'manifest.json'
    research.write_json(path, manifest)
    if report_failure:
        monkeypatch.setattr(image_attacker_ablation, 'report', lambda *args: (_ for _ in ()).throw(RuntimeError('report disk error')))
        with pytest.raises(RuntimeError, match='report disk error'):
            image_attacker_ablation.run(path, 'diagnostic')
    else:
        image_attacker_ablation.run(path, 'diagnostic')
    receipt = json.loads((root / 'diagnostic.json').read_text())
    metrics = json.loads((root / 'diagnostic_metrics.json').read_text())
    assert receipt['status'] == ('failed' if report_failure else 'completed')
    assert metrics['stochastic_steps_checked'] == 49
    assert metrics['max_replay_logprob_error'] == 0 and metrics['initial_gradient_norm'] > 0
    assert 'status' not in metrics and 'max_replay_logprob_error' not in receipt
