import json
from pathlib import Path

import pytest

from scripts.image_experiments import STAGES, report, submission_command
from ai_content_detector.rl_evasion.image_evasion.research import notebook_helpers


def test_production_jobs_use_regular_qos_and_safe_dependencies(tmp_path):
    manifest = {'directory': str(tmp_path), 'jobs': {}}
    for index, stage in enumerate(STAGES):
        command = submission_command(tmp_path / 'manifest.json', manifest, stage)
        assert '--qos=qos_gpu_h100-t3' in command
        assert '--hold' in command and '--requeue' in command
        dependencies = [item for item in command if item.startswith('--dependency=')]
        assert bool(dependencies) == bool(STAGES[stage])
        if dependencies:
            assert dependencies == ['--dependency=afterok:' + ':'.join(manifest['jobs'][name]['job_id'] for name in STAGES[stage])]
        manifest['jobs'][stage] = {'job_id': str(index + 100)}


def test_notebook_dataset_helpers_load_without_scanning_or_materializing_images():
    path = Path(__file__).resolve().parents[1] / 'notebooks/bench-imai-artifact.ipynb'
    helpers = notebook_helpers(path)
    assert 'build_binary_image_dataset' in helpers
    assert 'train_ds' not in helpers and 'X_all' not in helpers


def test_live_report_marks_pending_work_and_keeps_base_and_attack_separate(tmp_path):
    root, repo = tmp_path / 'run', tmp_path / 'repo'
    root.mkdir()
    (repo / 'docs').mkdir(parents=True)
    manifest = {'directory': str(root), 'repository': str(tmp_path), 'jobs': {}}
    (tmp_path / 'docs').mkdir()
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest))
    (root / 'measurements').mkdir()
    values = {'accuracy': 0.8, 'f1_macro': 0.8, 'roc_auc': 0.9,
              'human_false_positive_rate': 0.1, 'ai_detection_rate': 0.7, 'attack_success_rate': 0.3}
    for name in ('warmup_on_base', 'warmup_on_attacked', 'updated_on_base', 'updated_on_attacked'):
        (root / 'measurements' / f'{name}.json').write_text(json.dumps({'fixed_0.5': values}))
    report(path)
    text = (root / 'report.md').read_text()
    assert 'not submitted' in text
    assert 'updated_on_base' in text and 'updated_on_attacked' in text
    assert '0.10 reference similarity' in text
    assert 'No second attacker update' in text
    assert len((root / 'metrics.csv').read_text().splitlines()) == 5


def test_attach_defender_adopts_test_receipt_without_changing_parent(tmp_path, monkeypatch):
    from scripts import image_experiments as module
    from ai_content_detector.rl_evasion.image_evasion import research
    old, new = tmp_path / 'parent', tmp_path / 'new'
    (old / 'long').mkdir(parents=True)
    (old / 'warmup').mkdir()
    (old / 'warmup/completed.pt').write_bytes(b'warmed checkpoint')
    (old / 'partitions.json').write_text('{}')
    (old / 'long/effective_config.json').write_text(json.dumps({'num_train_epochs': 1600, 'evasion_mode': 'probability'}))
    parent = {'directory': str(old), 'partitions': str(old / 'partitions.json'),
              'partitions_sha256': research.file_hash(old / 'partitions.json'),
              'warmup_sha256': research.file_hash(old / 'warmup/completed.pt'),
              'resources': {}, 'models': {}, 'jobs': {'test': {'job_id': '1534163'}}}
    parent_path = old / 'manifest.json'
    parent_path.write_text(json.dumps(parent))
    new.mkdir()
    path = new / 'manifest.json'
    path.write_text(json.dumps({'directory': str(new), 'jobs': {}, 'protocol': {}}))
    monkeypatch.setattr(module, 'create_manifest', lambda: path)
    assert module.defender_manifest(parent_path) == path
    result = json.loads(path.read_text())
    assert json.loads(parent_path.read_text()) == parent
    assert result['submission_stages'] == ['defend']
    assert module.receipt_path(result, 'attack') == old / 'test.json'
    assert '--dependency=afterok:1534163' in module.submission_command(path, result, 'defend')
    assert (new / 'warmup').resolve() == old / 'warmup'


def test_shared_defender_workflow_reuses_identical_images_before_and_after(tmp_path, monkeypatch):
    from ai_content_detector.rl_evasion.image_evasion import research, ablation
    fake_fit, fake_eval, base = [{'path': 'fit'}], [{'path': 'attack'}], [{'path': 'base'}]
    real_fit, real_eval = [{'path': 'humanfit'}], [{'path': 'humantest'}]
    (tmp_path / 'warmup').mkdir()
    (tmp_path / 'warmup/threshold.json').write_text(json.dumps({'threshold': 0.12}))
    (tmp_path / 'partitions.json').write_text(json.dumps({'real_fitting': real_fit, 'real_evaluation': real_eval, 'evaluation': ['heldout']}))
    monkeypatch.setattr(ablation, 'defender_inputs', lambda *args: (fake_fit, fake_eval, base))
    monkeypatch.setattr(research, 'sd_pipeline', lambda *args: pytest.fail('The legacy sampler must not run'))
    calls, fits = [], []
    def phase(manifest, checkpoint, fake, real, name, stop, threshold):
        calls.append((checkpoint, fake, real, name, threshold))
        return {'image_identities': research.digest(fake + real)}
    def fit(manifest, initial, fake, real, destination, stop):
        fits.append((initial, fake, real, destination))
        return 'updated checkpoint'
    monkeypatch.setattr(research, 'phase_evaluation', phase)
    monkeypatch.setattr(research, 'fit_defender', fit)
    manifest = {'directory': str(tmp_path), 'attacker_parent_manifest': '/parent/manifest.json',
                'attacker_config': {'num_train_epochs': 1600}}
    research.defend(manifest, object())
    assert len(fits) == 1 and fits[0][1:3] == (fake_fit, real_fit)
    assert [call[3] for call in calls] == ['warmup_on_base', 'warmup_on_attacked', 'updated_on_attacked', 'updated_on_base']
    assert calls[0][1] is calls[3][1] and calls[1][1] is calls[2][1]
    assert all(call[2] == real_eval and call[4] == 0.12 for call in calls)
    assert json.loads((tmp_path / 'results.json').read_text())['saved_images_reused']


def test_generation_only_reuses_sampler_without_detector_queries(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    from PIL import Image
    from ai_content_detector.rl_evasion.image_evasion import ablation
    from ai_content_detector.rl_evasion.checkpointing import load_checkpoint
    prompts = [{'sample_id': str(i), 'prompt': str(i)} for i in range(8)]
    calls = []
    def sample(*args, **kwargs):
        calls.append(kwargs['num_inference_steps'])
        return torch.zeros(8, 1), [], []
    monkeypatch.setattr(ablation, 'pipeline_sample_with_logprob', sample)
    monkeypatch.setattr(ablation, 'decode_latents_to_pil', lambda *args: [Image.new('RGB', (4, 4)) for _ in range(8)])
    detector = SimpleNamespace(predict_batch=lambda *args: pytest.fail('No fitting-image detector queries'))
    trainer = SimpleNamespace(pipeline=SimpleNamespace(unet=SimpleNamespace(device='cpu', eval=lambda: None)),
                              reward_fn=SimpleNamespace(detector_rewards=[SimpleNamespace(detector=detector)],
                                                        _get_clip_score=lambda *args: pytest.fail('No fitting-image CLIP queries')),
                              _encode_prompts=lambda *args: (torch.zeros(8, 1), torch.zeros(8, 1)))
    stop = SimpleNamespace(requested=False)
    result = ablation.evaluate(trainer, prompts, tmp_path, stop, 'selected', score=False, seed=20261026)
    assert len(result['images']) == 8 and calls == [50]
    state = load_checkpoint(tmp_path / 'evaluation.pt')
    assert not state['scores'] and not state['clip'] and state['config']['score'] is False
    assert ablation.evaluate(trainer, prompts, tmp_path, stop, 'selected', score=False, seed=20261026) == result
    assert calls == [50]


def test_followup_report_contains_only_defender_comparison(tmp_path):
    from scripts import image_experiments as module
    (tmp_path / 'docs').mkdir()
    root = tmp_path / 'run'
    (root / 'measurements').mkdir(parents=True)
    values = {'ai_detection_rate': 0.9, 'attack_success_rate': 0.1, 'human_false_positive_rate': 0.02, 'roc_auc': 0.95}
    for phase in ['warmup_on_base', 'warmup_on_attacked', 'updated_on_base', 'updated_on_attacked']:
        (root / 'measurements' / f'{phase}.json').write_text(json.dumps({'fixed_0.5': values}))
    manifest = {'directory': str(root), 'repository': str(tmp_path), 'attacker_parent_manifest': '/parent/manifest.json',
                'jobs': {'defend': {'job_id': '999'}}, 'report_document': 'docs/defender.md'}
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest))
    module.report(path)
    text = (root / 'report.md').read_text()
    assert 'Base SD1.5' in text and 'Selected attacker' in text and 'Warmed' in text and 'Updated' in text
    assert 'pilot' not in text and 'Official DDA is evaluated' not in text


@pytest.mark.parametrize('parent_status,parent_job,allowed', [('completed', '123', True), ('failed', '123', False), ('completed', 'wrong', False)])
def test_defender_worker_validates_adopted_receipt_before_training(tmp_path, monkeypatch, parent_status, parent_job, allowed):
    from scripts import image_experiments as module
    from ai_content_detector.rl_evasion.image_evasion import research
    parent, root = tmp_path / 'parent', tmp_path / 'run'
    parent.mkdir()
    root.mkdir()
    (tmp_path / 'docs').mkdir()
    (parent / 'test.json').write_text(json.dumps({'status': parent_status, 'job_id': parent_job}))
    checkpoint = root / 'official.pt'
    checkpoint.write_bytes(b'official identity')
    manifest = {'directory': str(root), 'repository': str(tmp_path), 'code_sha256': {},
                'attacker_parent_manifest': str(parent / 'manifest.json'), 'report_document': 'docs/defender.md',
                'resources': {'checkpoint': {'files': {'DDA_ckpt.pth': {'path': str(checkpoint), 'sha256': research.file_hash(checkpoint)}}}},
                'jobs': {'attack': {'job_id': '123', 'adopted_from_directory': str(parent), 'receipt_stage': 'test'},
                         'defend': {'job_id': '456'}}}
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv('SLURM_JOB_ID', '456')
    calls = []
    monkeypatch.setattr(research, 'defend', lambda *args: calls.append(True))
    if allowed:
        module.run(path, 'defend')
        assert json.loads((root / 'defend.json').read_text())['status'] == 'completed'
        assert len(calls) == 1 and 'completed' in (root / 'report.md').read_text()
    else:
        with pytest.raises(ValueError, match='not complete|another job'):
            module.run(path, 'defend')
        assert not calls


def test_completed_dependency_requires_matching_accounting_and_receipt(tmp_path):
    from scripts import image_experiments as module
    (tmp_path / 'test.json').write_text(json.dumps({'status': 'completed', 'job_id': '123'}))
    job = {'job_id': '123', 'adopted_from_directory': str(tmp_path), 'receipt_stage': 'test',
           'completion_verified': {'job_id': '123', 'state': 'COMPLETED', 'exit_code': '0:0'}}
    manifest = {'directory': str(tmp_path), 'jobs': {'attack': job}}
    command = module.submission_command(tmp_path / 'manifest.json', manifest, 'defend')
    assert not any(arg.startswith('--dependency=') for arg in command)
    job['completion_verified']['state'] = 'FAILED'
    with pytest.raises(ValueError, match='proof and receipt'):
        module.submission_command(tmp_path / 'manifest.json', manifest, 'defend')


def test_evaluation_worker_runs_only_scoring_and_requests_one_hour(tmp_path, monkeypatch):
    from scripts import image_experiments as module
    from ai_content_detector.rl_evasion.image_evasion import research, ablation
    (tmp_path / 'docs').mkdir()
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint')
    manifest = {'directory': str(tmp_path), 'repository': str(tmp_path), 'code_sha256': {},
                'evaluation_checkpoint': str(checkpoint), 'report_document': 'docs/evaluation.md',
                'resources': {'checkpoint': {'files': {'DDA_ckpt.pth': {'path': str(checkpoint), 'sha256': research.file_hash(checkpoint)}}}},
                'jobs': {'evaluate': {'job_id': '123'}}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    calls = []
    monkeypatch.setattr(ablation, 'evaluate_checkpoint', lambda *args: calls.append(True))
    monkeypatch.setattr(research, 'attack', lambda *args: pytest.fail('No attacker training'))
    monkeypatch.setattr(research, 'defend', lambda *args: pytest.fail('No defender training'))
    module.run(path, 'evaluate')
    assert calls == [True]
    assert json.loads((tmp_path / 'evaluate.json').read_text())['status'] == 'completed'
    assert 'completed' in (tmp_path / 'report.md').read_text()
    command = module.submission_command(path, manifest, 'evaluate')
    assert '--time=01:00:00' in command and '--qos=qos_gpu_h100-t3' in command
    assert not any(arg.startswith('--dependency=') for arg in command)


def test_requested_checkpoint_test_reuses_baseline_and_never_trains(tmp_path, monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    import torch
    from ai_content_detector.rl_evasion.config import ImageEvasionConfig
    from ai_content_detector.rl_evasion.checkpointing import atomic_save
    from ai_content_detector.rl_evasion.image_evasion import research, ablation
    parent, root = tmp_path / 'parent', tmp_path / 'evaluation'
    (parent / 'warmup').mkdir(parents=True)
    (parent / 'test').mkdir()
    root.mkdir()
    warmup = parent / 'warmup/completed.pt'
    warmup.write_bytes(b'warmed identity')
    warm_hash = research.file_hash(warmup)
    data = {'fitting': [{'sample_id': str(i), 'prompt': str(i)} for i in range(1000)],
            'evaluation': [{'sample_id': str(i), 'prompt': str(i)} for i in range(1000, 1500)],
            'real_evaluation': [{'label': 0} for _ in range(500)]}
    research.write_json(parent / 'partitions.json', data)
    pm = {'directory': str(parent), 'partitions': str(parent / 'partitions.json'),
          'jobs': {'long': {'job_id': '100'}, 'test': {'job_id': '101'}}}
    research.write_json(parent / 'manifest.json', pm)
    for name in ['long', 'test']:
        research.write_json(parent / f'{name}.json', {'status': 'completed', 'job_id': pm['jobs'][name]['job_id']})
    config = ImageEvasionConfig()
    requested = parent / 'budget-12800.pt'
    atomic_save({'config': asdict(config), 'model': {'requested': torch.ones(1)}}, requested)
    signature = {'prompts': research.digest(data['evaluation']), 'checkpoint': 'initial_SD1.5',
                 'sampler': 'float32_trajectory_DDIM_eta1_CFG7.5', 'seed': 20261025}
    baseline = {'num_images': 500, 'evasion_rate': 0.012, 'clip_mean': 0.33, 'signature': signature}
    research.write_json(parent / 'test_results.json', {'initial': baseline})
    atomic_save({'config': {'checkpoint': warm_hash, 'rows': research.digest(data['real_evaluation'])},
                 'scores': [0.01] * 500}, parent / 'test/human.pt')
    baseline_hash = research.file_hash(parent / 'test_results.json')
    manifest = {'directory': str(root), 'checkpoint_parent_manifest': str(parent / 'manifest.json'),
                'evaluation_checkpoint': str(requested), 'evaluation_checkpoint_sha256': research.file_hash(requested),
                'baseline_test_sha256': baseline_hash, 'partitions_sha256': research.file_hash(parent / 'partitions.json'),
                'warmup_sha256': warm_hash, 'attacker_config': asdict(config)}
    trainer = SimpleNamespace(pipeline=SimpleNamespace(unet=object()))
    monkeypatch.setattr(research, 'image_trainer', lambda *args: trainer)
    restored = []
    monkeypatch.setattr(ablation, 'restore_model', lambda model, state: restored.append(state))
    def score(actual_trainer, prompts, directory, stop, identity):
        assert actual_trainer is trainer and prompts == data['evaluation']
        assert identity == manifest['evaluation_checkpoint_sha256']
        atomic_save({'scores': [0.9] * 5 + [0.1] * 495}, directory / 'evaluation.pt')
        return {'num_images': 500, 'evasion_rate': 0.99, 'clip_mean': 0.2}
    monkeypatch.setattr(ablation, 'evaluate', score)
    research.write_json(parent / 'warmup/threshold.json', {'threshold': 0.4})
    ablation.evaluate_checkpoint(manifest, SimpleNamespace(requested=False))
    result = json.loads((root / 'results.json').read_text())
    assert list(restored[0]) == ['requested']
    assert result['initial'] == baseline
    assert result['evaluated']['classification']['attack_success_rate'] == 0.99
    assert result['evaluated']['clip_retention_vs_initial'] == pytest.approx(0.2 / 0.33)
    assert result['post_hoc_evaluation'] and result['training_performed'] is False
    assert research.file_hash(parent / 'test_results.json') == baseline_hash
