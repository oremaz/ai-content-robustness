"""Evaluate a new generator with the frozen Qwen-warmed detector ensemble."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from translation_experiments import (activate_source, digest, read, save,
                                     score_condition, sha, test_rows)


def validate(config):
    if digest({k: v for k, v in config.items() if k != 'signature'}) != config['signature']:
        raise ValueError('Transfer manifest changed')
    directory = Path(config['directory'])
    for name, expected in config['frozen_hashes'].items():
        if sha(directory / name) != expected:
            raise ValueError('Frozen source or input changed: ' + name)
    data = read(directory / 'partitions.json')['text']
    rows = test_rows(config)
    if len(rows) != config['num_samples'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Incomplete or repeated transfer cohort')
    if {r['id'] for r in rows} & {r['id'] for split in ('train', 'validation') for r in data[split]}:
        raise ValueError('Transfer cohort overlaps detector adaptation data')
    if {r['human_text'] for r in rows} & {r['human_text'] for split in ('train', 'validation') for r in data[split]}:
        raise ValueError('Transfer human controls overlap detector adaptation data')


def transfer_prompts(tokenizer, rows):
    prompts = []
    for row in rows:
        formatted = tokenizer.apply_chat_template([{'role': 'user', 'content':
            'Continue the following news article. Return only the continuation, in English.\n\n'
            + row['question']}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt = formatted
        if tokenizer.encode(prompt) != tokenizer.encode(formatted, add_special_tokens=False) and tokenizer.bos_token:
            prompt = formatted.removeprefix(tokenizer.bos_token)
        if tokenizer.encode(prompt) != tokenizer.encode(formatted, add_special_tokens=False):
            raise ValueError('Chat prompt would duplicate or omit special tokens')
        prompts.append(prompt)
    return prompts


def generate(config):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    from ai_content_detector.rl_evasion.defender_comparison.generation import generate_batched

    rows = test_rows(config)
    path = Path(config['directory']) / 'base_en_test.json'
    if path.exists():
        saved = read(path)
        if saved['signature'] != config['signature'] or [r['id'] for r in saved['rows']] != [r['id'] for r in rows]:
            raise ValueError('Saved transfer outputs changed')
        return
    location = config['models']['generator']['path']
    tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True, padding_side='left')
    prompts = transfer_prompts(tokenizer, rows)
    model = AutoModelForMultimodalLM.from_pretrained(location, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    settings = config['sampling']
    attacker = SimpleNamespace(model=model, tokenizer=tokenizer, config=SimpleNamespace(
        generator_model=location, max_seq_length=2048, max_new_tokens=settings['max_tokens'],
        temperature=settings['temperature'], top_p=settings['top_p']))
    outputs = generate_batched(attacker, prompts, config['directory'],
        context={'manifest': config['signature'], 'prompt_format': 'native_chat_thinking_disabled'},
        seed=config['seed'], batch_size=settings['batch'])
    save(path, {'signature': config['signature'],
                'rows': [dict(id=r['id'], text=text) for r, text in zip(rows, outputs)]})
    print('Saved Gemma outputs:', len(outputs), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['phase'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('stage', choices=['generate', 'score'])
    args = parser.parse_args()
    config = read(args.manifest)
    validate(config)
    activate_source(config)
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
    from ai_content_detector.rl_evasion.config import seed_everything

    directory = Path(config['directory'])
    receipt = directory / 'receipts' / (args.stage + '.json')
    if receipt.exists() and read(receipt).get('signature') == config['signature']:
        return
    seed_everything(config['seed'])
    try:
        with StopRequest() as stop:
            if args.stage == 'generate':
                generate(config)
            else:
                score_condition(config, 'base', 'en', None, stop)
                metrics = read(directory / 'scores/base_en_metrics.json')
                save(directory / 'results.json', dict(generator=config['models']['generator'],
                    protocol=config['transfer_protocol'], metrics=metrics))
                print(metrics['ensemble_mean'], flush=True)
            if stop.requested:
                raise CheckpointPause('Transfer stage saved')
        save(receipt, dict(status='completed', signature=config['signature']))
    except CheckpointPause:
        raise SystemExit(75)


if __name__ == '__main__':
    main()
