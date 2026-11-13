"""Portable report counts and incomplete translation conditions."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def summary(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'tex/translation_summary.py'
    spec = importlib.util.spec_from_file_location('translation_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    (tmp_path / 'tex/generated').mkdir(parents=True)
    return module


def test_tables_distinguish_human_false_positives_and_scoring_coverage(summary, tmp_path):
    snapshot = {'gemma': {'metrics': {'ensemble_mean': {'num_ai': 128, 'num_human': 128,
        'ai_detection_rate': 1., 'human_false_positive_rate': 3/128, 'roc_auc': 1.}}},
        'groups': {'english-chinese-2k': {'conditions': {}}, 'french-welsh-200': {'n': 200,
        'conditions': {'textseal2_cy_gemma': {'ensemble_mean': {'num_ai': 199,
            'ai_detection_rate': 196/199}, 'watermark': {'detection_rate': 14/200}}}}}}
    summary.generate(snapshot)
    gemma = (tmp_path / 'tex/generated/generator_transfer.tex').read_text()
    assert 'Human continuations & 128 & 3 & 2.34\\%' in gemma
    assert 'Gemma-4-12B-it outputs & 128 & 128 & 100.00\\%' in gemma
    translation = (tmp_path / 'tex/generated/translation_french_welsh.tex').read_text()
    assert '98.49\\%$^{\\dagger}$' in translation
    assert '199/200 scorable' in translation and '7.00\\%' in translation
    assert 'Pending' in translation


def test_snapshot_rejects_cohort_leakage(summary, tmp_path):
    directory = tmp_path / 'translation'
    directory.mkdir()
    (directory / 'partitions.json').write_text(json.dumps({'text': {
        'train': [{'id': 'overlap'}], 'validation': [], 'test': [{'id': 'overlap'}]}}))
    with pytest.raises(ValueError, match='overlaps'):
        summary.export(directory, tmp_path / 'gemma')


def test_snapshot_rejects_unexplained_sample_loss(summary, tmp_path):
    directory = tmp_path / 'translation'
    scores = directory / 'english-chinese-2k/scores'
    scores.mkdir(parents=True)
    (directory / 'partitions.json').write_text(json.dumps({'text': {
        'train': [], 'validation': [], 'test': [{'id': 'held-out', 'human_text': 'human'}]}}))
    (scores / 'base_zh-Hans_index_metrics.json').write_text(json.dumps({
        'ensemble_mean': {'num_ai': 1999, 'num_human': 2000}}))
    with pytest.raises(ValueError, match='sample counts changed'):
        summary.export(directory, tmp_path / 'gemma')
