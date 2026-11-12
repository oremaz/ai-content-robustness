"""Regenerate the paired ensemble cohort with the same forbidden sequences."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def forbidden_ids(tokenizer, strings):
    return sorted({tuple(tokenizer.encode(prefix + text, add_special_tokens=False))
                   for text in strings for prefix in ('', ' ', '\n')})


def write_tables(root, original, constrained=None):
    lines = []
    for title, result in [('Previous comparison', original), ('New comparison', constrained)]:
        if result:
            protocol = result.get('protocol', {})
            title += f" ({protocol.get('num_samples', 500)} samples, {protocol.get('regime', 'unrestricted')})"
        lines += [f'## {title}', '', '| Metric | Before GRPO | After GRPO | Human |',
                  '|---|---:|---:|---:|']
        for label, key in [('Generation perplexity', 'gen_ppl'), ('Mean token entropy (bits)', 'mean_token_entropy_bits'),
                           ('E5 similarity', 'mean_semantic_similarity'), ('MAUVE', 'mauve')]:
            values = [f'{result["metrics"][group][key]:.4f}' if result and key in result['metrics'][group]
                      else ('pending' if result is None else 'N/A') for group in ('before', 'after', 'human')]
            lines.append('| ' + ' | '.join([label, *values]) + ' |')
        lines.append('')
    (root / 'comparison.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    root = args.manifest.parent
    config = json.loads(args.manifest.read_text())
    for relative, expected in config['source_hashes'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source changed: ' + relative)
    for path, expected in config['generation_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Generation input changed: ' + path)
    from unittest.mock import patch
    with patch('importlib.metadata.packages_distributions', return_value=config['package_map']):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    from ai_content_detector.rl_evasion.defender_comparison.generation import generate_batched

    original_base = json.loads(Path(config['original_base']).read_text())
    rows = original_base['evaluation']
    expected = config.get('num_samples', 500)
    if len(rows) != expected:
        raise ValueError(f'Expected {expected} prompts')
    tokenizer = AutoTokenizer.from_pretrained(config['generator'], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bad = forbidden_ids(tokenizer, config['forbidden_strings'])
    if any(not ids for ids in bad):
        raise ValueError('Empty forbidden token sequence')
    (root / 'forbidden_ids.json').write_text(json.dumps([list(ids) for ids in bad], indent=2) + '\n')
    model = AutoModelForCausalLM.from_pretrained(config['generator'], local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    model.generation_config.bad_words_ids = [list(ids) for ids in bad] or None
    if config['forbidden_strings']:
        from forbidden_text import ForbiddenTextProcessor
        processor = ForbiddenTextProcessor(tokenizer, config['forbidden_strings'])
        model.generate = partial(model.generate, logits_processor=[processor])
    outputs = {}
    for name in ('before', 'after'):
        if name == 'after':
            model = PeftModel.from_pretrained(model, config['adapter'], is_trainable=False).eval()
            model.generation_config.bad_words_ids = [list(ids) for ids in bad] or None
        attacker = SimpleNamespace(model=model, tokenizer=tokenizer, config=SimpleNamespace(
            generator_model=config['generator'], **config['sampling']))
        generated = generate_batched(attacker, [r['prompt'] for r in rows], root / name,
                                     context={'group': name, 'forbidden_ids': bad, 'constraint_version': config.get('constraint_version', 'token_sequences_v1'),
                                              'adapter_hash': config['generation_hashes'][str(Path(config['adapter'])/'adapter_model.safetensors')]},
                                     seed=42, batch_size=config.get('generation_batch_size', 8))
        if any(text in output for output in generated for text in config['forbidden_strings']):
            raise ValueError('Forbidden string survived decoding')
        outputs[name] = [dict(row, ai=text) for row, text in zip(rows, generated)]
        path = root / (name + '_pairs.json')
        path.write_text(json.dumps({'evaluation': outputs[name]} if name == 'before' else outputs[name], indent=2) + '\n')
        print(name, len(generated), config.get('regime', 'constrained'), 'outputs saved', flush=True)
    del attacker, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    evaluation = dict(config)
    evaluation['inputs'] = {name: str(root / (name + '_pairs.json')) for name in outputs}
    evaluation['input_hashes'] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for name, path in evaluation['inputs'].items()}
    manifest = root / 'evaluation_manifest.json'
    manifest.write_text(json.dumps(evaluation, indent=2) + '\n')
    subprocess.run([sys.executable, str(root / 'compare_grpo_quality.py'), str(manifest)], check=True)
    previous = Path(config['original_results'])
    write_tables(root, json.loads(previous.read_text()) if previous.exists() else None,
                 json.loads((root / 'results.json').read_text()))


if __name__ == '__main__':
    main()
