"""Native prompt tokenization and held-out controls for generator transfer."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def driver(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('transfer_worker', scripts / 'evaluate_generator_transfer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('automatic_bos', [False, True])
def test_native_template_has_exactly_one_bos(driver, automatic_bos):
    class Tokenizer:
        bos_token = '<bos>'

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['enable_thinking'] is False
            assert 'in English' in messages[0]['content']
            return '<bos>native chat prompt'

        def encode(self, text, add_special_tokens=True):
            return ([2] if text.startswith('<bos>') else []) + ([2] if add_special_tokens and automatic_bos else []) + [7]

    tokenizer = Tokenizer()
    prompt = driver.transfer_prompts(tokenizer, [{'question': 'news prefix'}])[0]
    assert tokenizer.encode(prompt) == [2, 7]


@pytest.mark.parametrize('overlap', ['id', 'human_text'])
def test_transfer_rejects_adaptation_overlap(driver, tmp_path, overlap):
    train = {'id': 'train', 'human_text': 'training human'}
    test = {'id': 'test', 'human_text': 'test human', 'question': 'news prefix'}
    test[overlap] = train[overlap]
    driver.save(tmp_path / 'partitions.json', {'text': {'train': [train], 'validation': [], 'test': [test]}})
    config = {'directory': str(tmp_path), 'num_samples': 1, 'frozen_hashes': {}}
    config['signature'] = driver.digest(config)
    with pytest.raises(ValueError, match='overlap'):
        driver.validate(config)
