"""Freeze fresh test articles for unrestricted and constrained GRPO comparisons."""

import hashlib
import json
from pathlib import Path
import runpy
import shutil
from datetime import datetime, timezone
from unittest.mock import patch


def main():
    root = Path(__file__).resolve().parents[1]
    parent = root / 'results/research_workflow/grpo_quality/constrained-20261111T232811Z/manifest.json'
    config = json.loads(parent.read_text())
    original = root / 'results/research_workflow/grpo_quality/20261111T230402Z/results.json'
    directory = root / 'results/research_workflow/grpo_quality' / ('test5k-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    directory.mkdir()
    with patch('importlib.metadata.packages_distributions', return_value=config['package_map']):
        from datasets import Dataset
        from transformers import AutoTokenizer
    dataset_path = Path('/lustre/fswork/projects/rech/mbm/ual37ci/.cache/huggingface/datasets/abisee___cnn_dailymail/3.0.0/0.0.0/96df5e686bee6baa90b8bee7c28b81fa3fa6223d/cnn_dailymail-test.arrow')
    dataset = Dataset.from_file(str(dataset_path))
    prior_path = root / 'results/research_workflow/defender_comparison/prompt_manifest.json'
    prior = json.loads(prior_path.read_text())
    excluded = set(prior['train']) | set(prior['evaluation'])
    tokenizer = AutoTokenizer.from_pretrained(config['generator'], local_files_only=True)
    rows, seen = [], set()
    for index in range(len(dataset)):
        article = dataset[index]['article']
        words = article.split()
        if len(words) < 70:
            continue
        prompt, reference = ' '.join(words[:50]), ' '.join(words[50:])
        if prompt in excluded or prompt in seen:
            continue
        seen.add(prompt)
        human = tokenizer.decode(tokenizer.encode(reference, add_special_tokens=False)[:256], skip_special_tokens=True)
        rows.append({'sample_id': hashlib.sha256(prompt.encode()).hexdigest(), 'prompt': prompt,
                     'reference': reference, 'human': human, 'ai': '', 'dataset_index': index,
                     'article_sha256': hashlib.sha256(article.encode()).hexdigest()})
        if len(rows) == 5000:
            break
    if len(rows) != 5000:
        raise ValueError('Not enough disjoint test prompts')
    cohort = directory / 'cohort.json'
    cohort.write_text(json.dumps({'evaluation': rows}, indent=2) + '\n')
    receipt = {'source': str(dataset_path), 'split': 'test', 'num_samples': len(rows),
               'selection': 'first 5000 eligible unique test prompts, excluding all recorded attacker train/evaluation prompts',
               'excluded_prompt_count': len(excluded), 'prompt_manifest_sha256': hashlib.sha256(prior_path.read_bytes()).hexdigest(),
               'cohort_sha256': hashlib.sha256(cohort.read_bytes()).hexdigest()}
    (directory / 'cohort_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    helper = runpy.run_path(str(root / 'scripts/generate_constrained_grpo.py'))
    for regime in ('unrestricted', 'constrained'):
        out = directory / regime
        out.mkdir()
        cfg = dict(config, num_samples=5000, regime=regime, constraint_version='literal_text_v2', generation_batch_size=32, extra_mauve_seeds=[43, 44],
                   original_base=str(cohort), original_results=str(original) if regime == 'unrestricted' else str(directory/'unrestricted/results.json'),
                   human_reference='Same 5000 held-out CNN/DailyMail test articles; human continuation after first 50 words, capped at 256 Qwen tokens, fixed across models and regimes',
                   forbidden_strings=[] if regime == 'unrestricted' else ['<think>', '</think>', 'OPTIONS:'])
        cfg['generation_hashes'] = {p:h for p,h in cfg['generation_hashes'].items()
                                    if p != config['original_base'] and p != config['original_results']}
        cfg['generation_hashes'][str(cohort)] = receipt['cohort_sha256']
        shutil.copytree(root/'ai_content_detector',out/'source/ai_content_detector',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.ipynb'))
        for name in ('generate_constrained_grpo.py','compare_grpo_quality.py','run_constrained_grpo.slurm', 'forbidden_text.py'):
            shutil.copyfile(root/'scripts'/name,out/name)
        cfg['source_hashes'] = {str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in out.rglob('*') if p.is_file()}
        (out/'manifest.json').write_text(json.dumps(cfg,indent=2)+'\n')
        helper['write_tables'](out,json.loads(original.read_text()))
    (directory/'historical_500.md').write_text((directory/'unrestricted/comparison.md').read_text().split('## New comparison')[0])
    print(directory, flush=True)


if __name__ == '__main__':
    main()
