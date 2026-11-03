"""Discrete-channel gradients and inference contracts for the latent text experiment."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from ai_content_detector.watermarking.latent_text import (
    LatentEmbedder, TextExtractor, channel, embedded, pack_tokens, straight_through_logits,
)
from ai_content_detector.watermarking.protocol import DEFAULT_STAGES, LATENT_STAGES, STAGES
from ai_content_detector.watermarking.text import forward_kl


def test_sweep_calibration_and_validation_are_disjoint():
    path = Path(__file__).resolve().parents[2] / 'scripts/latent_sweep_validation.py'
    spec = importlib.util.spec_from_file_location('latent_sweep_validation_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calibration, evaluation = module.validation_partitions([{'id': str(i)} for i in range(8)])
    assert [row['id'] for row in calibration] == ['0', '1', '2', '3']
    assert [row['id'] for row in evaluation] == ['4', '5', '6', '7']
    calibration, evaluation = module.validation_partitions([{'id': x} for x in ('a', 'b', 'c', 'a')])
    assert {row['id'] for row in calibration}.isdisjoint(row['id'] for row in evaluation)
    assert sum(row['id'] == 'a' for row in calibration) == 2
    with pytest.raises(ValueError, match='disjoint'):
        module.validation_partitions([{'id': 'same'}, {'id': 'same'}])


def test_student_selection_uses_validation_detection_then_quality():
    path = Path(__file__).resolve().parents[2] / 'scripts/latent_sweep_selection.py'
    spec = importlib.util.spec_from_file_location('latent_sweep_selection_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def candidate(rate, perplexity, n=128):
        return {'evaluation_documents': n, 'against_base': {'detection': {'rate': rate}},
                'quality': {'reference_generation_perplexity': perplexity}}
    values = {'baseline': candidate(.4, 3), 'first': candidate(.5, 5), 'second': candidate(.5, 4)}
    assert module.select_candidate(values) == 'second'
    values['second']['evaluation_documents'] = 64
    with pytest.raises(ValueError, match='sizes differ'):
        module.select_candidate(values)


def test_teacher_selection_can_keep_baseline_without_training(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    path = scripts/'latent_sweep_train_selected.py'
    spec = importlib.util.spec_from_file_location('latent_sweep_train_selected_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def sha(p):
        import hashlib
        return hashlib.sha256(p.read_bytes()).hexdigest()
    candidates = {}
    for label, rate in [('baseline', .6), ('strength_070', .5)]:
        receipt, result = tmp_path/f'{label}_receipt.json', tmp_path/f'{label}_result.json'
        receipt.write_text(json.dumps({'status': 'completed'}))
        result.write_text(json.dumps({'split': 'validation only', 'evaluation_documents': 128,
                                      'against_base': {'detection': {'rate': rate}, 'false_positive': {'rate': 0}},
                                      'quality': {'reference_generation_perplexity': 4}}))
        candidates[label] = {'receipt': str(receipt), 'results': str(result)}
    config = {'directory': str(tmp_path), 'script_sha256': sha(path),
              'selection_helper': str(scripts/'latent_sweep_selection.py'),
              'selection_helper_sha256': sha(scripts/'latent_sweep_selection.py'), 'candidates': candidates}
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    module.run(config_path)
    assert json.loads((tmp_path/'teacher_selection.json').read_text())['selected'] == 'baseline'


def test_teacher_selection_cli_starts():
    path = Path(__file__).resolve().parents[2]/'scripts/latent_sweep_train_selected.py'
    result = subprocess.run([sys.executable, str(path), '--help'], capture_output=True, text=True)
    assert result.returncode == 0


def test_dense_distillation_is_forward_kl_with_valid_token_normalization():
    torch.manual_seed(12)
    student = torch.randn(2, 3, 13, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    valid = torch.tensor([[True, False, True], [False, True, False]])
    target = teacher.detach().softmax(-1)
    expected = F.kl_div(student[valid].log_softmax(-1), target[valid], reduction="sum") / valid.sum()
    actual = forward_kl(student, teacher, valid, chunk=2)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert teacher.grad is None
    assert student.grad[~valid].abs().sum() == 0
    torch.testing.assert_close(student.grad[valid], (student.detach().softmax(-1)[valid]-target[valid])/valid.sum())
    with pytest.raises(ValueError):
        forward_kl(student, teacher, torch.zeros_like(valid))


def test_embedder_starts_at_identity_and_bounds_hidden_perturbation():
    torch.manual_seed(13)
    embedder = LatentEmbedder(12, 4, width=8, strength=.35)
    hidden, message = torch.randn(3, 7, 12), torch.randint(2, (3, 4))
    torch.testing.assert_close(embedder(hidden, message), hidden)
    with torch.no_grad():
        embedder.up.weight.normal_()
    changed = embedder(hidden, message)
    rms = hidden.square().mean(-1, keepdim=True).sqrt()
    assert ((changed-hidden).abs() <= .35*rms + 1e-6).all()
    assert not torch.allclose(changed, embedder(hidden, 1-message))


def test_straight_through_forward_is_hard_text_and_extractor_gradient_is_hard_only():
    torch.manual_seed(14)
    extractor = TextExtractor(19, 4, width=8)
    ids, mask = pack_tokens([[2, 5, 7], [3]], "cpu")
    soft = torch.randn(2, 5, 8, requires_grad=True)
    soft_mask = torch.ones(2, 5, dtype=torch.bool)
    hard = extractor(ids, mask)
    hard_grads = torch.autograd.grad(hard.sum(), tuple(extractor.parameters()))
    actual = straight_through_logits(extractor, ids, mask, soft, soft_mask)
    torch.testing.assert_close(actual, hard.detach())
    actual.sum().backward()
    assert soft.grad.abs().sum() > 0
    for parameter, expected in zip(extractor.parameters(), hard_grads):
        torch.testing.assert_close(parameter.grad, expected)


def test_extractor_is_padding_invariant_and_has_no_message_input():
    extractor = TextExtractor(19, 4, width=8).eval()
    ids, mask = pack_tokens([[2, 5, 7]], "cpu")
    padded = F.pad(ids, (0, 6), value=17)
    padded_mask = F.pad(mask, (0, 6), value=False)
    torch.testing.assert_close(extractor(ids, mask), extractor(padded, padded_mask))


def test_categorical_channel_trains_embedder_through_frozen_head():
    torch.manual_seed(15)
    embedder = LatentEmbedder(12, 4, 8)
    extractor = TextExtractor(19, 4, 8)
    head = nn.Linear(12, 19, bias=False).requires_grad_(False)
    hidden = torch.randn(2, 7, 12)
    message = torch.randint(2, (2, 4)).float()
    sampled, soft, kl = channel(embedder, extractor, head, hidden, message, chunk=3)
    torch.testing.assert_close(kl, torch.zeros(()), atol=1e-6, rtol=0)
    assert sampled.shape == (2, 7) and sampled.dtype == torch.long
    assert ((sampled >= 0) & (sampled < 19)).all()
    logits = straight_through_logits(extractor, sampled, torch.ones_like(sampled).bool(), soft,
                                     torch.ones(2, 7, dtype=torch.bool))
    F.binary_cross_entropy_with_logits(logits, message).backward()
    assert embedder.up.weight.grad.abs().sum() > 0
    assert head.weight.grad is None
    assert extractor.embedding.weight.grad.abs().sum() > 0


def test_external_hook_is_removed_after_failure():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(12, 19, bias=False)
        def get_output_embeddings(self):
            return self.head
    model = Model()
    embedder = LatentEmbedder(12, 4, 8)
    with torch.no_grad():
        embedder.up.bias.fill_(.2)
    hidden = torch.randn(2, 7, 12)
    baseline = model.head(hidden).detach()
    with pytest.raises(RuntimeError):
        with embedded(model, embedder, [0, 1, 0, 1]):
            assert not torch.allclose(model.head(hidden), baseline)
            raise RuntimeError("interrupt")
    torch.testing.assert_close(model.head(hidden), baseline)


def test_latent_jobs_are_gated_and_preserve_original_stage_set():
    assert len(DEFAULT_STAGES) == 12
    assert not set(DEFAULT_STAGES) & set(LATENT_STAGES)
    assert STAGES["latent_distill"] == ["latent_teacher"]
    assert STAGES["latent_teacher"] == ["latent_embedder"]
    assert STAGES["latent_clean"] == ["latent_evaluate"]
