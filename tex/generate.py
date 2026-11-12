"""Regenerate manuscript summary tables from the retained portable metrics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    source = ROOT.parent / 'docs/watermarking/data/metrics.json'
    data = json.loads(source.read_text())
    rows = []
    labels = {'original':'Original marked','clean':'Clean control','vanilla':'Ordinary rewrite',
              'bira':'BIRA','direct_topk':'Top-10 resampling','smoothing':'Smoothing'}
    for scheme, values in data['text'].items():
        if scheme not in ('textseal2','latent'):
            continue
        name = 'TextSeal 2' if scheme == 'textseal2' else 'Text student'
        for method, result in values['detection']['conditions'].items():
            if method not in labels:
                continue
            op = result['operating_points']['fpr001']
            access = "O" if method in ("vanilla", "bira") else "P" if method in ("direct_topk", "smoothing") else "--"
            rows.append(f"{name} & {labels[method]} & {access} & {op['detected']}/{result['n']} & {100*op['detection_rate']:.1f}\\% \\\\")
    (ROOT/'generated/text_summary.tex').write_text(
        '\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lllrr}\\toprule\n'
        'Scheme & Condition & Access & Detected & Rate\\\\\\midrule\n'+'\n'.join(rows)+
        '\n\\bottomrule\\end{tabular}\n\\caption{Text-watermark experiments with Qwen3.5-4B instruct on 256 HC3 questions per condition. TextSeal 2 is the native keyed sampler; Text student is the local distilled latent-text adaptation. Detected is the number still flagged by each frozen owner test, and Rate is that count divided by 256. TextSeal 2 uses native $p\\leq0.001$; the local student uses extractor score $>-0.6263284683$. Ordinary rewriting and BIRA use Llama-3.1-8B-Instruct on a saved marked answer. Top-10 resampling and smoothing generate fresh answers under simulated probability access; TextSeal 2 exposes pre-watermark probabilities, while the student exposes its marked model probabilities. Access: O means saved outputs only; P means a simulated probability-only target API without target weight access; -- marks a control. Initial detection differs, so post-attack detection alone does not count removals from initially detected sources.}\\end{table}\n')
    rows=[]
    for model, values in data['images'].items():
        for method,label in [('weight_unmodified_supplied','Marked original'),('weight_clean_lora_2500','Clean LoRA 2,500'),('weight_clean_full_2500','Clean full 2,500'),('output_nfpa_40','NFPA 40')]:
            result=values['quality']['conditions'][method]
            quality=result['quality']['all']
            access = "O" if method.startswith("output_") else "--" if method == "weight_unmodified_supplied" else "W"
            rows.append(f"{model.upper()} & {label} & {access} & {100*result['detection']['rate']:.1f}\\% & {quality['lpips']['mean']:.4f} & {quality['clip_alignment']['mean']:.4f} \\\\")
    (ROOT/'generated/image_summary.tex').write_text(
        '\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lllrrr}\\toprule\n'
        'Model & Condition & Access & Detected & LPIPS & Class CLIP\\\\\\midrule\n'+'\n'.join(rows)+
        '\n\\bottomrule\\end{tabular}\n\\caption{DistSeal image-watermark removal on RAR-XL at 256 pixels and UViT-H/DC-AE at 512 pixels, with the original clean decoder and 512 matched samples per condition. Quality means come from the later matched native-resolution audit, consistently across weight and output interventions. Detected is the percentage retaining owner detection after the named intervention. Strict bit-accuracy thresholds are $>0.640625$ for RAR and $>0.6875$ for DC-AE. Clean LoRA and full training use 2,500 clean-data updates; NFPA strength 40 transforms the source output using independent SD2.1. Access: W requires the marked generator weights and clean training data; O requires only saved target outputs; -- marks the baseline. LPIPS is mean candidate-to-source perceptual distance, with lower values indicating closer pairs. Class CLIP is mean candidate alignment to the requested ImageNet class using CLIP ViT-B/32; it measures class agreement and does not certify retention of the original scene.}\\end{table}\n')

if __name__ == '__main__':
    main()
