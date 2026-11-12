"""Compare fixed GRPO texts with frozen Mistral likelihoods and E5 similarity."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def entropy(ids):
    counts = Counter(ids)
    return -sum((n / len(ids)) * math.log2(n / len(ids)) for n in counts.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    from unittest.mock import patch
    with patch('importlib.metadata.packages_distributions', return_value=config['package_map']):
        import torch
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import mauve
    from ai_content_detector.rl_evasion.text_evasion.rewards import SemanticSimilarityReward

    root = args.manifest.parent
    inputs = {}
    for name, path in config['inputs'].items():
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != config['input_hashes'][name]:
            raise ValueError('Input changed: ' + name)
        inputs[name] = json.loads(data)
    before, after = inputs['before']['evaluation'], inputs['after']
    expected = config.get('num_samples', 500)
    if len(before) != expected or len(after) != expected:
        raise ValueError(f'Expected {expected} paired samples')
    if any(any(a[k] != b[k] for k in ('sample_id', 'prompt', 'reference')) for a, b in zip(before, after)):
        raise ValueError('Sample alignment changed')
    groups = {'before': before, 'after': after,
              'human': [dict(r, ai=r['human']) for r in before]}
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    summaries, features = {}, {}
    with torch.inference_mode():
        for name, rows in groups.items():
            scores, vectors = [], []
            for i, row in enumerate(rows):
                prompt = tokenizer.encode(row['prompt'], add_special_tokens=False)
                tokens = tokenizer.encode(row['ai'], add_special_tokens=False)
                if not prompt or not tokens:
                    raise ValueError('Empty prompt or completion')
                ids = torch.tensor([prompt + tokens], device='cuda')
                if ids.shape[1] > model.config.max_position_embeddings:
                    raise ValueError('Context overflow')
                output = model(ids, use_cache=False, output_hidden_states=True)
                logits = output.logits[0, len(prompt)-1:-1].float()
                targets = ids[0, len(prompt):]
                nll = torch.nn.functional.cross_entropy(logits, targets, reduction='sum').item()
                vectors.append(output.hidden_states[-1][0, -1].float().cpu().numpy())
                scores.append({'sample_id': row['sample_id'], 'nll': nll, 'tokens': len(tokens),
                               'token_entropy_bits': entropy(tokens)})
                del output, logits
                if (i + 1) % 50 == 0:
                    print(name, i + 1, flush=True)
            features[name] = np.stack(vectors)
            summaries[name] = {'gen_ppl': math.exp(sum(r['nll'] for r in scores) / sum(r['tokens'] for r in scores)),
                               'mean_token_entropy_bits': float(np.mean([r['token_entropy_bits'] for r in scores]))}
            (root / (name + '_scores.json')).write_text(json.dumps(scores, indent=2) + '\n')
            np.save(root / (name + '_features.npy'), features[name])
    del model
    torch.cuda.empty_cache()
    encoder = SemanticSimilarityReward(config['e5'], device='cuda')
    for name in ('before', 'after'):
        rows = groups[name]
        values = encoder.batch([r['reference'] for r in rows], [r['ai'] for r in rows])
        summaries[name]['mean_semantic_similarity'] = float(np.mean(values))
        (root / (name + '_semantic.json')).write_text(json.dumps(values) + '\n')
        result = mauve.compute_mauve(p_features=features['human'], q_features=features[name],
                                    num_buckets=25, seed=42, verbose=False)
        summaries[name]['mauve'] = float(result.mauve)
        summaries[name]['mauve_by_seed'] = {'42': float(result.mauve)}
        for seed in config.get('extra_mauve_seeds', []):
            repeated = mauve.compute_mauve(p_features=features['human'], q_features=features[name],
                                           num_buckets=25, seed=seed, verbose=False)
            summaries[name]['mauve_by_seed'][str(seed)] = float(repeated.mauve)
    result = {'metrics': summaries, 'protocol': config, 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(),
              'definitions': {'gen_ppl': 'exp(total completion NLL / total completion tokens), conditioned on prompt; separate prompt/completion tokenization, no BOS',
                              'entropy': 'Mean per-document empirical Mistral token unigram entropy in bits',
                              'mauve': 'Custom frozen Mistral final-token last-layer features of prompt+completion; 25 buckets, seed 42; human continuations as recorded in input pairs (see manifest human_reference)',
                              'semantic': 'Existing E5 query-prefix mean-pool cosine against full human reference, max length 512'}}
    (root / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
