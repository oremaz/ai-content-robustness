"""Post-hoc joint evasion and preservation diagnostics for retained text rewrites."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TEXT=ROOT/'results/research_workflow/text_instruct/20261103T231534057688Z'
JINA=ROOT/'results/research_workflow/jina_similarity/20261104T081330240270Z'


def indexed(path):
    value=json.loads(path.read_text())
    rows=value if isinstance(value,list) else value['rows']
    values={r['id']:r for r in rows}
    if len(values)!=len(rows):raise ValueError('Duplicate sample IDs')
    return values


def preservation(row):
    return (row['nli_label']=='entailment' and row['reverse_nli_label']=='entailment'
        and not row['nli_truncated'] and row['numeric_set_f1']==1.
        and row['mistral_perplexity'] is not None and math.isfinite(row['mistral_perplexity'])
        and row['mistral_perplexity']>0
        and not row['invalid_perplexity'] and not row['wrapper_leakage'] and not row['long_letter_run'])


def main():
    results=[];sources={}
    def read(path):
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return indexed(path)
    for scheme in ['textseal2','latent']:
        folder=TEXT/scheme
        detection_path=folder/'detection.json'
        sources[str(detection_path.relative_to(ROOT))]=hashlib.sha256(detection_path.read_bytes()).hexdigest()
        threshold=json.loads(detection_path.read_text())['thresholds']['fpr001']
        def detected(row):
            return row['eligible'] and (row['score']>=threshold if scheme=='textseal2' else row['score']>threshold)
        originals=read(folder/'original_scores.json')
        for method in ['vanilla','bira']:
            after=read(folder/(method+'_scores.json'));quality=read(folder/(method+'_quality.json'))
            cosine=read(JINA/scheme/(method+'_similarity.json'))
            if any(set(x)!=set(originals) for x in [after,quality,cosine]):raise ValueError('Unaligned preservation/detection IDs')
            for minimum in [0.,.85,.9,.95]:
                positives=[i for i in originals if detected(originals[i])]
                removed=[i for i in positives if not detected(after[i])]
                passes=[i for i in originals if preservation(quality[i]) and cosine[i]['cosine']>=minimum]
                joint=set(removed)&set(passes)
                results.append({'scheme':scheme,'method':method,'minimum_jina_cosine':minimum,
                    'n':len(originals),'originally_detected':len(positives),'raw_removed_from_positives':len(removed),
                    'preservation_proxy_pass':len(passes),'joint_removed_and_preservation_pass':len(joint),
                    'joint_rate_all_outputs':len(joint)/len(originals),
                    'joint_rate_original_positives':len(joint)/len(positives)})
    output={'status':'Post-hoc diagnostic on previously inspected retained data; not a prespecified confirmation or a guarantee of factual fidelity.',
            'preservation_proxy':'Bidirectional NLI entailment, no NLI truncation, numeric-set F1 exactly one, valid independent perplexity, no wrapper leakage or long letter runs; plus stated Jina cosine floor.',
            'sources':sources,'results':results}
    (ROOT/'tex/provenance/quality_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['\\begin{table}[ht]\\centering\\small','\\begin{tabular}{llrrrr}\\toprule',
           'Scheme & Rewrite & \\shortstack{Initially\\\\detected} & Removed & \\shortstack{Text-rule\\\\pass} & Joint\\\\\\midrule']
    for r in results:
        if r['minimum_jina_cosine']==.9:
            label='TextSeal 2' if r['scheme']=='textseal2' else 'Text student'
            lines.append(f"{label} & {('Ordinary' if r['method']=='vanilla' else 'BIRA')} & {r['originally_detected']} & {r['raw_removed_from_positives']} & {r['preservation_proxy_pass']} & {r['joint_removed_and_preservation_pass']} \\\\")
    lines+=['\\bottomrule\\end{tabular}',
        '\\caption{Post-hoc paired text-watermark audit on Qwen3.5-4B instruct, 256 source answers per row. TextSeal 2 uses native $p\\leq0.001$; the local distilled student uses extractor score $>-0.6263284683$. Initially detected counts marked sources passing that test. Removed counts their attacked answers that become undetected. Text-rule pass counts all rewrites satisfying bidirectional NLI entailment without truncation, exact numeric-set agreement, valid positive independent Mistral perplexity, no wrapper leakage or long letter runs, and Jina cosine $\\geq0.90$. Joint counts the intersection of removal and text-rule pass. Ordinary and BIRA rewriting both use independent Llama-3.1-8B-Instruct with output-only target access. These counts use automatic proxies and are not human annotations.}\\end{table}']
    (ROOT/'tex/generated/quality_qualified_text.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps([r for r in results if r['minimum_jina_cosine']==.9],indent=2))

if __name__=='__main__':main()
