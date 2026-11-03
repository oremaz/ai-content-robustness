"""Distill one teacher chosen on independent validation, then validate the student."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from latent_sweep_selection import select_candidate


def save_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != config['script_sha256']:
        raise ValueError('Teacher-selection script changed')
    if hashlib.sha256(Path(config['selection_helper']).read_bytes()).hexdigest() != config['selection_helper_sha256']:
        raise ValueError('Selection helper changed')
    selection_path = Path(config['directory'])/'teacher_selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
    else:
        results, hashes = {}, {}
        for label, item in config['candidates'].items():
            receipt, result_path = Path(item['receipt']), Path(item['results'])
            if not receipt.exists() or json.loads(receipt.read_text()).get('status') != 'completed' or not result_path.exists():
                continue
            raw = result_path.read_bytes()
            result = json.loads(raw)
            if result.get('split') != 'validation only' or result.get('mode', 'teacher') != 'teacher':
                raise ValueError('Candidate is not a teacher validation result')
            results[label] = result
            hashes[label] = hashlib.sha256(raw).hexdigest()
        selected = select_candidate(results)
        selection = {'selection_rule': 'highest independent validation detection against base; ties use lower reference perplexity',
                     'selected': selected, 'validation_sha256': hashes,
                     'validation': {name: {'detection': item['against_base']['detection'],
                                           'false_positive': item['against_base']['false_positive'],
                                           'quality': item['quality']} for name, item in results.items()},
                     'excluded': sorted(set(config['candidates'])-set(results)),
                     'note': 'Stronger teachers failed their original bit-accuracy gate; this separate validation selects on detection.'}
        save_json(selection_path, selection)
    label = selection['selected']
    if label == 'baseline':
        return
    item = config['candidates'][label]
    manifest_path = Path(item['manifest'])
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest['source'])
    sys.path.insert(0, str(source))
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
    from ai_content_detector.rl_evasion.config import seed_everything
    from ai_content_detector.watermarking import latent_text
    from ai_content_detector.watermarking.protocol import file_hash, use_sources, write_json
    for name, expected in manifest['code_sha256'].items():
        if file_hash(source/name) != expected:
            raise ValueError('Selected teacher source changed')
    if file_hash(manifest['data']) != manifest['data_sha256']:
        raise ValueError('Selected teacher partitions changed')
    job = os.environ['SLURM_JOB_ID']
    entry = manifest['jobs'].get('latent_distill', {})
    if entry.get('job_id') != job:
        manifest['jobs']['latent_distill'] = {'job_id': job, 'selection': str(selection_path),
                                               'previous_job': entry,
                                               'note': 'selected by independent validation after preliminary gate failure'}
        save_json(manifest_path, manifest)
        os.chmod(manifest_path, 0o600)
    receipt = Path(manifest['directory'])/'latent_distill.json'
    complete = receipt.exists() and json.loads(receipt.read_text()).get('status') == 'completed'
    use_sources(manifest)
    seed_everything(manifest['seed'])
    if not complete:
        try:
            write_json(receipt, {'status': 'running', 'job_id': job})
            with StopRequest() as stop:
                latent_text.fit(manifest, stop)
            write_json(receipt, {'status': 'completed', 'job_id': job})
        except CheckpointPause:
            write_json(receipt, {'status': 'paused', 'job_id': job})
            subprocess.run(['scontrol', 'requeue', job], check=True)
            return
        except BaseException as error:
            write_json(receipt, {'status': 'failed', 'job_id': job, 'error': repr(error)})
            raise
    validation_config = item['student_validation_config']
    validation = json.loads(Path(validation_config).read_text())
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
    os.execve(sys.executable, [sys.executable, validation['script'], validation_config], environment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    run(parser.parse_args().config)
