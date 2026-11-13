"""Export portable translation/transfer evidence and regenerate report tables."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'tex/provenance/translation_transfer_results.json'
MODELS = [('base', 'Clean'), ('textseal2', 'TextSeal 2'), ('latent', 'Text student')]
TRANSLATORS = [('index', 'Index'), ('gemma', 'TranslateGemma')]


def export(translation, gemma):
    hashes = {}
    def read(path):
        content = path.read_bytes()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    cohorts = read(translation / 'partitions.json')['text']
    ids = {name: {r['id'] for r in rows} for name, rows in cohorts.items()}
    if ids['test'] & (ids['train'] | ids['validation']):
        raise ValueError('Test cohort overlaps detector adaptation data')
    references = {name: {r['human_text'] for r in rows} for name, rows in cohorts.items()}
    if references['test'] & (references['train'] | references['validation']):
        raise ValueError('Test human references overlap detector adaptation data')
    groups = {}
    for name, n in [('english-chinese-2k', 2000), ('french-welsh-200', 200)]:
        directory = translation / name
        conditions = {p.stem.removesuffix('_metrics'): read(p)
                      for p in sorted((directory / 'scores').glob('*_metrics.json'))}
        for label, result in conditions.items():
            metric = result['ensemble_mean']
            coverage = result.get('ensemble_scoring', {})
            if metric['num_ai'] != coverage.get('n_scorable', n) or metric['num_human'] != n:
                raise ValueError('Metric sample counts changed: ' + label)
            if coverage.get('n_generated', n) != n:
                raise ValueError('Attempted sample count changed: ' + label)
            if result.get('watermark', {}).get('n', n) != n:
                raise ValueError('Watermark sample count changed: ' + label)
        expected = 9 if n == 2000 else 12
        groups[name] = dict(n=n, conditions=conditions, complete=len(conditions) == expected,
                            completed_conditions=len(conditions), expected_conditions=expected)
        manifest = directory / ('manifest-v2.json' if (directory / 'manifest-v2.json').exists() else 'manifest.json')
        config = read(manifest)
        groups[name]['protocol'] = dict(signature=config['signature'], seed=config['seed'],
            sampling=config['sampling'], translation_batch_size=config['translation_batch_size'],
            models={k: {field: model[field] for field in ('name', 'revision')}
                    for k, model in config['models'].items()},
            detector_checkpoints=read(directory / 'warmup_checkpoints.json'))
        latent = read(Path(config['latent_manifest']))
        groups[name]['protocol']['local_text_student'] = dict(protocol_sha256=latent['protocol_sha256'],
            watermark_training=latent['latent_training'], distillation=latent['text_training'],
            stop_step=config['distillation_stop_step'])
        if (directory / 'mauve.json').exists():
            groups[name]['mauve'] = read(directory / 'mauve.json')
    result = read(gemma / 'results.json')
    for stage in ('generate', 'score'):
        if read(gemma / 'receipts' / (stage + '.json'))['status'] != 'completed':
            raise ValueError('Gemma evaluation is incomplete')
    snapshot = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        warmup=dict(generator='Qwen/Qwen3.5-9B instruct', dataset='CNN/DailyMail 3.0.0',
                    pair_pool=len(cohorts['train']), validation_pool=len(cohorts['validation']),
                    test_pool=len(cohorts['test']), requested_language='English',
                    language_purity_filter=False, test_id_overlap=0),
        gemma=dict(generator={k: result['generator'][k] for k in ('name', 'revision')},
                   protocol=result['protocol'], metrics=result['metrics']),
        groups=groups, source_hashes=hashes)
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + '\n')
    return snapshot


def rate(value):
    return 'N/A' if value is None else f'{100 * value:.2f}\\%'


def table(path, header, rows, caption):
    path.write_text('\\begin{table}[ht]\\centering\\small\n'
        '\\begin{adjustbox}{max width=\\linewidth}\n'
        '\\begin{tabular}{' + 'l' * len(header) + '}\\toprule\n'
        + ' & '.join(header) + '\\\\\\midrule\n'
        + '\n'.join(' & '.join(row) + '\\\\' for row in rows)
        + '\n\\bottomrule\\end{tabular}\\end{adjustbox}\n'
        + '\\caption{' + caption + '}\\label{tab:' + path.stem + '}\\end{table}\n')


def generate(snapshot):
    generated = ROOT / 'tex/generated'
    metric = snapshot['gemma']['metrics']['ensemble_mean']
    rows = []
    for name, n, positive_rate in [('Gemma-4-12B-it outputs', metric['num_ai'], metric['ai_detection_rate']),
                                  ('Human continuations', metric['num_human'], metric['human_false_positive_rate'])]:
        rows.append([name, str(n), str(round(n * positive_rate)), rate(positive_rate)])
    table(generated / 'generator_transfer.tex', ['Held-out class', '$N$', 'Flagged AI', 'Rate'], rows,
        'Zero-shot generator transfer within CNN/DailyMail news. The unchanged Qwen-instruct-warmed '
        'seven-member ensemble flags scores at least 0.5. The human row is a false-positive rate, '
        'not a detection success rate. No Gemma-12B fitting or threshold adjustment is performed. '
        'ROC AUC on these 256 passages is ' + f"{metric['roc_auc']:.2f}" + '.')
    conditions = snapshot['groups']['english-chinese-2k']['conditions']
    rows = []
    for arm, model in MODELS:
        for translator, label in TRANSLATORS:
            result = conditions.get(f'{arm}_zh-Hans_{translator}')
            wm = result.get('watermark', {}) if result else {}
            rows.append([model, label, rate(result['ensemble_mean']['ai_detection_rate']) if result else 'Pending',
                         rate(wm.get('detection_rate')), rate(wm.get('lost_given_source_detected'))])
    table(generated / 'translation_chinese.tex', ['Version', 'Translator', 'AI detection', 'WM detected', 'WM loss$^*$'], rows,
        'Chinese-requested generation followed by translation into English, 2,000 outputs per condition. '
        'AI detection uses the frozen ensemble at 0.5; WM detected is the translated watermark detection rate. '
        '$^*$WM loss conditions on a detected source watermark: 1,989 TextSeal 2 sources and 308 text-student sources. '
        'Loss is not a content-preserving removal rate. Unwatermarked controls have no WM test.')
    group = snapshot['groups']['french-welsh-200']
    conditions = group['conditions']
    rows, restricted = [], []
    for arm, model in MODELS:
        for language, name in [('fr', 'French'), ('cy', 'Welsh')]:
            values = []
            watermarks = []
            source_rate = None
            for translator, label in TRANSLATORS:
                result = conditions.get(f'{arm}_{language}_{translator}')
                value = rate(result['ensemble_mean']['ai_detection_rate']) if result else 'Pending'
                if result and result['ensemble_mean']['num_ai'] != group['n']:
                    value += '$^{\\dagger}$'
                    restricted.append(f"{model}/{name}/{label}: {result['ensemble_mean']['num_ai']}/200 scorable")
                values.append(value)
                watermarks.append(rate(result.get('watermark', {}).get('detection_rate')) if result else 'Pending')
                if result and 'source_detected' in result.get('watermark', {}):
                    source_rate = result['watermark']['source_detected'] / group['n']
            rows.append([model, name, rate(source_rate), *values, *watermarks])
    note = (' $\\dagger$' + '; '.join(restricted) + '.') if restricted else ''
    table(generated / 'translation_french_welsh.tex',
        ['Version', 'Source', 'Source WM', 'AI Index', 'AI Gemma', 'WM Index', 'WM Gemma'], rows,
        'French/Welsh pilot: 200 attempted outputs per condition; Gemma means TranslateGemma. '
        'Source WM is measured before translation. AI columns use the same frozen seven-member '
        'ensemble and 0.5 threshold. Watermark columns '
        'retain all 200 outputs. Texts below the released DivEye minimum of 15 words are unscorable '
        'for the full ensemble and are not successful valid evasions.' + note)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--translation-root', type=Path)
    parser.add_argument('--gemma-run', type=Path)
    args = parser.parse_args()
    if bool(args.translation_root) != bool(args.gemma_run):
        parser.error('Both local run directories are required for a snapshot refresh')
    snapshot = (export(args.translation_root.resolve(), args.gemma_run.resolve()) if args.translation_root
                else json.loads(SNAPSHOT.read_text()))
    generate(snapshot)
    print({k: v['complete'] for k, v in snapshot['groups'].items()})


if __name__ == '__main__':
    main()
