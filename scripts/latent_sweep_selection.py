"""Freeze a validation-only student choice, then evaluate its held-out test set."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def select_candidate(candidates):
    if not candidates:
        raise ValueError('No student validation completed')
    sizes = {item['evaluation_documents'] for item in candidates.values()}
    if len(sizes) != 1:
        raise ValueError('Student validation sizes differ')
    return max(sorted(candidates), key=lambda name: (
        candidates[name]['against_base']['detection']['rate'],
        -candidates[name]['quality']['reference_generation_perplexity']))


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != config['script_sha256']:
        raise ValueError('Selection script changed')
    output = Path(config['directory'])
    path = output/'selection.json'
    if path.exists():
        selection = json.loads(path.read_text())
    else:
        candidates, hashes = {}, {}
        for label, item in config['candidates'].items():
            receipt, results = Path(item['receipt']), Path(item['results'])
            if not receipt.exists() or json.loads(receipt.read_text()).get('status') != 'completed' or not results.exists():
                continue
            raw = results.read_bytes()
            result = json.loads(raw)
            if result.get('split') != 'validation only' or result.get('mode') != 'student':
                raise ValueError('Candidate is not a student validation result')
            candidates[label] = result
            hashes[label] = hashlib.sha256(raw).hexdigest()
        selected = select_candidate(candidates)
        selection = {'selection_rule': 'highest validation detection against base; ties use lower reference perplexity',
                     'selected': selected, 'validation_sha256': hashes,
                     'validation': {name: {'detection': item['against_base']['detection'],
                                           'false_positive': item['against_base']['false_positive'],
                                           'quality': item['quality']} for name, item in candidates.items()},
                     'excluded': sorted(set(config['candidates'])-set(candidates))}
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(selection, indent=2)+'\n')
        os.replace(temporary, path)
    selected = selection['selected']
    item = config['candidates'][selected]
    manifest = json.loads(Path(item['manifest']).read_text())
    if selected == 'baseline':
        if not (Path(manifest['directory'])/'latent_evaluate/results.json').exists():
            raise ValueError('Baseline held-out results are missing')
        return
    source = Path(manifest['source'])
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
    os.execve(sys.executable, [sys.executable, str(source/'scripts/watermark_experiments.py'),
                               'run', item['manifest'], 'latent_evaluate'], environment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    run(parser.parse_args().config)
