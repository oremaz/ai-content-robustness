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
            rows.append(f"{name} & {labels[method]} & {op['detected']}/{result['n']} & {100*op['detection_rate']:.1f}\\% \\\\")
    (ROOT/'generated/text_summary.tex').write_text(
        '\\begin{table}[t]\\centering\\small\n\\begin{tabular}{llrr}\\toprule\n'
        'Scheme & Condition & Detected & Rate\\\\\\midrule\n'+'\n'.join(rows)+
        '\n\\bottomrule\\end{tabular}\n\\caption{Retained text detections at distinct primary operating points. Probability arms generate fresh responses; the student is a local adaptation.}\\end{table}\n')
    rows=[]
    for model, values in data['images'].items():
        for method,label in [('weight_unmodified_supplied','Marked original'),('weight_clean_lora_2500','Clean LoRA 2,500'),('weight_clean_full_2500','Clean full 2,500'),('output_nfpa_40','NFPA 40')]:
            result=values['quality']['conditions'][method]
            quality=result['quality']['all']
            rows.append(f"{model.upper()} & {label} & {100*result['detection']['rate']:.1f}\\% & {quality['lpips']['mean']:.4f} & {quality['clip_alignment']['mean']:.4f} \\\\")
    (ROOT/'generated/image_summary.tex').write_text(
        '\\begin{table}[t]\\centering\\small\n\\begin{tabular}{llrrr}\\toprule\n'
        'Model & Condition & Detected & LPIPS & Class CLIP\\\\\\midrule\n'+'\n'.join(rows)+
        '\n\\bottomrule\\end{tabular}\n\\caption{Matched image measurements, 512 outputs per condition. LPIPS measures paired drift; class CLIP is an incomplete utility proxy.}\\end{table}\n')

if __name__ == '__main__':
    main()
