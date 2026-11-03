"""Compare latent teachers on disjoint calibration and validation prompts."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
from ai_content_detector.rl_evasion.config import seed_everything
from ai_content_detector.watermarking.latent_text import embedded, generation_config, load_pair, score
from ai_content_detector.watermarking.protocol import detection_metrics, file_hash, operating_point, use_sources, write_json
from ai_content_detector.watermarking.text import generate, generation_perplexity, load_model


def validation_partitions(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['id'], []).append(row)
    calibration, evaluation = [], []
    for group in groups.values():
        (calibration if len(calibration) < len(rows)//2 else evaluation).extend(group)
    if not calibration or not evaluation:
        raise ValueError('Calibration and validation prompts must be nonempty and disjoint')
    return calibration, evaluation


def run(path):
    config = json.loads(Path(path).read_text())
    if file_hash(__file__) != config['script_sha256']:
        raise ValueError('Validation script changed')
    manifest = json.loads(Path(config['watermark_manifest']).read_text())
    if config['source'] != manifest['source']:
        raise ValueError('Validation source differs from experiment')
    for name, expected in manifest['code_sha256'].items():
        if file_hash(Path(config['source']) / name) != expected:
            raise ValueError('Frozen experiment source changed')
    if file_hash(manifest['data']) != manifest['data_sha256']:
        raise ValueError('Validation data changed')
    output = Path(config['directory'])
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('status') == 'completed':
        return
    use_sources(manifest)
    seed_everything(manifest['seed'])
    try:
        write_json(receipt, {'status': 'running'})
        with StopRequest() as stop:
            rows = json.loads(Path(manifest['data']).read_text())['text']['validation']
            calibration, evaluation = validation_partitions(rows)
            model, tokenizer = load_model(manifest['qwen']['path'])
            embedder, extractor, selected = load_pair(manifest)
            config_generation = generation_config(manifest)
            base_cal = generate(model, tokenizer, calibration, output/'base_calibration.json', config_generation, stop)
            base_eval = generate(model, tokenizer, evaluation, output/'base_validation.json', config_generation, stop)
            mode = config.get('mode', 'teacher')
            if mode == 'teacher':
                with embedded(model, embedder, manifest['latent_training']['message']):
                    candidate = generate(model, tokenizer, evaluation, output/'candidate_validation.json', config_generation, stop)
            elif mode == 'student':
                student, _ = load_model(Path(manifest['directory'])/manifest['distilled_model'])
                candidate = generate(student, tokenizer, evaluation, output/'candidate_validation.json', config_generation, stop)
            else:
                raise ValueError(f'Unknown latent validation mode: {mode}')
            message = manifest['latent_training']['message']
            human_cal = [tokenizer.decode(row['tokens'][50:306]) for row in calibration]
            human_eval = [tokenizer.decode(row['tokens'][50:306]) for row in evaluation]
            human_cal_scores, _ = score(extractor, tokenizer, human_cal, message)
            base_cal_scores, _ = score(extractor, tokenizer, [r['text'] for r in base_cal], message)
            threshold = max(operating_point(human_cal_scores), operating_point(base_cal_scores))
            candidate_scores, _ = score(extractor, tokenizer, [r['text'] for r in candidate], message)
            human_scores, _ = score(extractor, tokenizer, human_eval, message)
            base_scores, _ = score(extractor, tokenizer, [r['text'] for r in base_eval], message)
            quality = generation_perplexity(model, tokenizer, candidate)
            quality['mean_generated_tokens'] = float(np.mean([len(r['tokens']) for r in candidate]))
            results = {'split': 'validation only', 'calibration_documents': len(calibration),
                       'evaluation_documents': len(evaluation), 'selected_step': selected['step'],
                       'mode': mode, 'threshold': threshold,
                       'against_human': detection_metrics(candidate_scores, human_scores, threshold),
                       'against_base': detection_metrics(candidate_scores, base_scores, threshold), 'quality': quality,
                       'config': manifest['latent_training']}
            write_json(output/'results.json', results)
        write_json(receipt, {'status': 'completed'})
    except CheckpointPause:
        write_json(receipt, {'status': 'paused'})
        subprocess.run(['scontrol', 'requeue', os.environ['SLURM_JOB_ID']], check=True)
    except BaseException as error:
        write_json(receipt, {'status': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    run(parser.parse_args().config)
