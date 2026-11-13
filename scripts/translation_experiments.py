"""Frozen Qwen instruct translation study using the existing detector and watermark workers."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / 'results/research_workflow/watermarking/resources'
MODELS = {'generator': ('Qwen/Qwen3.5-9B', 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'),
          'index': ('IndexTeam/Index-Translate-9B', '4e06279e62352dab96bbeb6fa0579bca5e822d75'),
          'gemma': ('google/translategemma-12b-it', 'd1b225e1caa17f1ddc7e62065d8637d0923f34e2')}
LANGUAGES = {'en': 'English', 'zh-Hans': 'Simplified Chinese', 'fr': 'French', 'cy': 'Welsh'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def translation_messages(translator, language, text):
    if language not in LANGUAGES or language == 'en':
        raise ValueError('Expected a declared non-English source language')
    if translator == 'gemma':
        return [{'role': 'user', 'content': [{'type': 'text', 'source_lang_code': language,
                 'target_lang_code': 'en', 'text': text}]}]
    if translator != 'index':
        raise ValueError(translator)
    names = {'zh-Hans': '简体中文', 'fr': '法语', 'cy': '威尔士语'}
    return [{'role': 'user', 'content': f'请将以下{names[language]}文本翻译为英语，直接输出翻译结果，不要进行任何解释。\n\n{text}'}]


def check_cohorts(groups):
    seen, references = set(), set()
    for name, rows in groups.items():
        ids = [r['id'] for r in rows]
        texts = [r['human_text'] for r in rows]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f'Prompt overlap in {name}')
        if len(set(texts)) != len(texts) or references.intersection(texts):
            raise ValueError(f'Human reference overlap in {name}')
        seen.update(ids)
        references.update(texts)


def latent_training_partitions(data, length, batch):
    groups=dict(data['text'])
    for split in ('train','validation'):
        groups[split]=[r for r in groups[split] if len(r['tokens'])-r['prompt_length']>=length]
        if len(groups[split])<batch:
            raise ValueError('Insufficient full-length latent '+split+' responses')
    return dict(data,text=groups)


def latent_manifest(config):
    return read(config.get('latent_manifest',str(Path(config['directory'])/'latent_manifest.json')))


def prepare(args):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from ai_content_detector.rl_evasion.defender_comparison.plan import specifications
    from ai_content_detector.watermarking.instruct_text import chat_prompt
    models = {k: dict(name=n, revision=r, path=snapshot_download(n, revision=r, local_files_only=True, ignore_patterns=['*.pdf','*.md','*.png','*.jpg','*.gguf']))
              for k, (n, r) in MODELS.items()}
    tokenizer = AutoTokenizer.from_pretrained(models['generator']['path'], local_files_only=True)
    original = read(args.warmup)
    test = read(args.cohort)['evaluation']
    if len(test) != 5000:
        raise ValueError('MAUVE requires exactly 5000 fixed held-out examples')
    groups = {}
    seen_ids, seen_humans = set(), set()
    for split, rows in [('train', original['warmup']), ('validation', original['evaluation']), ('test', test)]:
        groups[split] = []
        for row in rows:
            instruction = 'Continue the following news article. Return only the continuation, in English.\n\n' + row['prompt']
            prompt = chat_prompt(tokenizer, instruction)
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            human = tokenizer.encode(row['reference'], add_special_tokens=False)[:256]
            human_text=tokenizer.decode(human, skip_special_tokens=True)
            if row['sample_id'] in seen_ids or human_text in seen_humans:
                if split == 'test':
                    raise ValueError('Fixed 5000-sample cohort overlaps or repeats human references')
                continue
            seen_ids.add(row['sample_id'])
            seen_humans.add(human_text)
            groups[split].append(dict(id=row['sample_id'], question=row['prompt'], prompt=prompt,
                prompt_length=len(prefix), tokens=prefix+human, human_text=human_text))
    check_cohorts(groups)
    print('Unique cohorts:', {k:len(v) for k,v in groups.items()}, flush=True)
    directory = ROOT / 'results/research_workflow/translation' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True)
    source = directory / 'source'
    shutil.copytree(ROOT/'ai_content_detector', source/'ai_content_detector', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.ipynb'))
    (source/'scripts').mkdir()
    for name in ('translation_experiments.py', 'run_translation.slurm', 'watermark_transport_train.py', 'watermark_survival_pilot.py'):
        shutil.copyfile(ROOT/'scripts'/name, source/'scripts'/name)
    shutil.copytree(RESOURCES/'textseal2/textseal', source/'textseal', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    save(directory/'partitions.json', {'text': groups})
    template = read(args.quality_template)
    plan = {'specifications': specifications(read(args.selection), directory)}
    latent = dict(directory=str(directory/'latent_training'), source=str(source), method='distseal_latent_text',
        qwen=models['generator'], data=str(directory/'partitions.json'), seed=42,
        distillation_stage='latent_distill', distilled_model='latent_distill/model',
        latent_training=dict(steps=7500, validate_every=500, batch=2, length=128, bits=32, width=256,
            strength=.65, temperature=1., lr=1e-4, kl_budget=.15, minimum_bit_accuracy=.5,
            message=__import__('numpy').random.default_rng(42).integers(2, size=32).tolist()),
        text_training=dict(steps=8000, clean_steps=1500, accumulation=8, length=256, lr=3e-5,
            clean_lr=1e-5, lora_rank=128, mode='lora'))
    Path(latent['directory']).mkdir()
    latent['protocol_sha256'] = digest(latent)
    save(directory/'latent_partitions.json',latent_training_partitions({'text':groups},
        latent['latent_training']['length'],latent['latent_training']['batch']))
    latent['data']=str(directory/'latent_partitions.json')
    latent['protocol_sha256']=digest({k:v for k,v in latent.items() if k!='protocol_sha256'})
    save(directory/'latent_manifest.json', latent)
    config = dict(directory=str(directory), source=str(source), models=models, seed=42, plan=plan,
        sampling=dict(batch=4, max_tokens=256, temperature=1., top_p=.95),
        mauve=dict(model=template['model'], buckets=25, seeds=[42,43,44]),
        runtime_archive=template['runtime_archive'], package_map=template['package_map'],
        inputs={str(Path(p).resolve()): sha(p) for p in (args.warmup,args.cohort,args.selection,args.quality_template)},
        frozen_hashes={str(p.relative_to(directory)): sha(p) for p in source.rglob('*') if p.is_file()},
        max_requeues=16,
        scope='Seven-member warmed ensemble; TextSeal2 sampler and local DistSeal-inspired text student. Stage III deferred.')
    config['frozen_hashes'].update({p.name: sha(p) for p in (directory/'partitions.json',directory/'latent_manifest.json',directory/'latent_partitions.json')})
    stages = ['preflight', 'generate:base:en:train', 'generate:base:en:validation',
              'generate:base:en:test', 'warmup', 'score:base:en', 'teacher', 'teacher_probe', 'student',
              'generate:textseal2:en:test', 'score:textseal2:en', 'generate:latent:en:test', 'score:latent:en', 'quality']
    for arm in ('base', 'textseal2', 'latent'):
        for lang in ('zh-Hans','fr','cy'):
            stages += [f'generate:{arm}:{lang}:test', f'source_score:{arm}:{lang}']
            for translator in ('index','gemma'):
                stages += [f'translate:{arm}:{lang}:{translator}', f'score:{arm}:{lang}:{translator}']
    save(directory/'stages.json', stages)
    config['frozen_hashes']['stages.json'] = sha(directory/'stages.json')
    config['signature'] = digest(config)
    save(directory/'manifest.json', config)
    print(directory/'manifest.json', flush=True)


def rows_file(config, name, expected):
    path = Path(config['directory'])/(name+'.json')
    if not path.exists():
        return path, []
    payload = read(path)
    rows = payload['rows']
    if payload['signature'] not in [config['signature'], *config.get('compatible_row_signatures', [])] or [r['id'] for r in rows] != [r['id'] for r in expected[:len(rows)]]:
        raise ValueError('Restart inputs or protocol changed')
    return path, rows


def retained(config, path, rows, stop):
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
    save(path, {'signature': config['signature'], 'rows': rows})
    if stop.requested:
        raise CheckpointPause('Completed batch saved')


def sample_limit(config):
    count=config.get('num_samples',5000)
    if type(count) is not int or not 1<=count<=5000:
        raise ValueError('Expected 1 to 5000 evaluation samples')
    return count


def test_rows(config):
    return read(Path(config['directory'])/'partitions.json')['text']['test'][:sample_limit(config)]


def record_throughput(config, name, records):
    save(Path(config['directory'])/'performance'/(name+'.json'),records)


def samples(config, arm, lang, split='test', translator=None):
    name = f'{arm}_{lang}_{split}' + (f'_{translator}' if translator else '')
    rows=read(Path(config['directory'])/(name+'.json'))['rows']
    return rows[:sample_limit(config)] if split=='test' else rows


def generate(config, arm, lang, split, stop):
    import torch
    from ai_content_detector.watermarking.text import load_model
    from ai_content_detector.watermarking.instruct_text import chat_prompt, textseal2_config
    from textseal.watermarking.generator import build_generator
    data = read(Path(config['directory'])/'partitions.json')['text'][split]
    path, rows = rows_file(config, f'{arm}_{lang}_{split}', data)
    limit=sample_limit(config) if split=='test' else len(data)
    if len(rows)>=limit:
        return
    latent = latent_manifest(config)
    model_path = str(Path(latent['directory'])/latent['distilled_model']) if arm == 'latent' else config['models']['generator']['path']
    model, tokenizer = load_model(model_path)
    wm = textseal2_config() if arm == 'textseal2' else replace(textseal2_config(), watermark_type='none')
    sampler = build_generator(model, tokenizer, wm)
    settings = config['sampling']
    batch_size=settings['batch']
    start=len(rows)
    timings=[]
    while start<limit:
        batch=data[start:min(start+batch_size,limit)]
        prompts=[chat_prompt(tokenizer,'Continue the following news article. Return only the continuation, in '
            +LANGUAGES[lang]+'.\n\n'+r['question']) for r in batch]
        torch.manual_seed(config['seed']+start)
        torch.cuda.reset_peak_memory_stats()
        began=time.perf_counter()
        outputs=None
        try:
            with torch.inference_mode():
                outputs=sampler.generate(prompts,max_gen_len=settings['max_tokens'],temperature=settings['temperature'],
                    top_p=settings['top_p'],return_ids=True)
        except torch.cuda.OutOfMemoryError:
            if batch_size==1:
                raise
        if outputs is None:
            import gc
            batch_size=max(1,batch_size//2)
            gc.collect();torch.cuda.empty_cache()
            print('Reducing generation batch after OOM:',batch_size,flush=True)
            continue
        elapsed=time.perf_counter()-began
        for row,ids in zip(batch,outputs):
            rows.append(dict(id=row['id'],text=tokenizer.decode(ids,skip_special_tokens=True),
                generated_tokens=len(ids),length_capped=len(ids)==settings['max_tokens'],
                generation_batch_size=len(batch),generation_seed=config['seed']+start))
        timings.append(dict(start=start,n=len(batch),seconds=elapsed,tokens=sum(map(len,outputs)),
            peak_memory_bytes=torch.cuda.max_memory_allocated(),batch_size=batch_size))
        retained(config,path,rows,stop)
        record_throughput(config,path.stem,timings)
        start=len(rows)
        print(path.name,start,'batch',batch_size,'texts/sec',round(len(batch)/elapsed,3),flush=True)


def translate(config, arm, lang, translator, stop):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    data = samples(config, arm, lang)
    if len(data) != sample_limit(config):
        raise ValueError('Incomplete source generation')
    path, rows = rows_file(config, f'{arm}_{lang}_test_{translator}', data)
    location = config['models'][translator]['path']
    tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    cls = AutoModelForImageTextToText if translator == 'gemma' else AutoModelForCausalLM
    model = cls.from_pretrained(location, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    batch_size = config.get('translation_batch_size',config['sampling']['batch'])
    start=len(rows)
    timings=[]
    while start<len(data):
        batch=data[start:start+batch_size]
        prompts=[tokenizer.apply_chat_template(translation_messages(translator,lang,r['text']),
            tokenize=False,add_generation_prompt=True,**({'enable_thinking':False} if translator=='index' else {})) for r in batch]
        inputs=tokenizer(prompts,padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
        torch.cuda.reset_peak_memory_stats()
        began=time.perf_counter();ids=None
        try:
            with torch.inference_mode():
                ids=model.generate(**inputs,do_sample=False,max_new_tokens=1024)[:,inputs.input_ids.shape[1]:]
        except torch.cuda.OutOfMemoryError:
            if batch_size==1:
                raise
        if ids is None:
            import gc
            del inputs
            batch_size=max(1,batch_size//2)
            gc.collect();torch.cuda.empty_cache()
            print('Reducing translation batch after OOM:',batch_size,flush=True)
            continue
        decoded=tokenizer.batch_decode(ids,skip_special_tokens=True)
        for row,tokens,text in zip(batch,ids,decoded):
            rows.append(dict(id=row['id'],text=text,source_sha256=digest(row),translation_batch_size=len(batch),
                length_capped=bool(tokens[-1]!=tokenizer.eos_token_id) and len(tokens)==1024))
        elapsed=time.perf_counter()-began
        timings.append(dict(start=start,n=len(batch),seconds=elapsed,peak_memory_bytes=torch.cuda.max_memory_allocated(),batch_size=batch_size))
        retained(config,path,rows,stop)
        record_throughput(config,path.stem,timings)
        start=len(rows)
        print(path.name,start,'batch',batch_size,'texts/sec',round(len(batch)/elapsed,3),flush=True)


def warmup(config, stop):
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    directory = Path(config['directory'])
    data = read(directory/'partitions.json')['text']
    pairs = {}
    for name, split in [('warmup','train'),('evaluation','validation')]:
        generated = samples(config,'base','en',split)
        if [r['id'] for r in generated] != [r['id'] for r in data[split]]:
            raise ValueError('Incomplete warmup generation')
        pairs[name] = [dict(sample_id=r['id'], prompt=r['question'], reference=r['human_text'], human=r['human_text'],
            ai=g['text'], eligible=len(g['text'].split())>=20 and len(r['human_text'].split())>=20)
            for r,g in zip(data[split],generated)]
    save(directory/'base_pairs.json',pairs)
    workflow.OUTPUT = directory
    workflow.warmup(config['plan'],stop)


def watermark_scores(config, arm, rows, label, stop):
    from transformers import AutoTokenizer
    from ai_content_detector.watermarking.instruct_text import textseal2_config, textseal2_scores
    from ai_content_detector.watermarking.latent_text import load_pair, score
    path, records = rows_file(config,label,rows)
    tokenizer = AutoTokenizer.from_pretrained(config['models']['generator']['path'],local_files_only=True)
    if arm == 'textseal2':
        from textseal.watermarking.detector import build_detector
        detector = build_detector(tokenizer,textseal2_config())
        scorer = lambda texts: textseal2_scores(detector,texts)
    else:
        latent = latent_manifest(config)
        embedder, extractor, _ = load_pair(latent)
        del embedder
        scorer = lambda texts: score(extractor,tokenizer,texts,latent['latent_training']['message'])[1]
    for start in range(len(records),len(rows),32):
        batch=rows[start:start+32]
        values=scorer([r['text'] for r in batch])
        records.extend(dict(v,id=r['id']) for r,v in zip(batch,values))
        retained(config,path,records,stop)
    return records


def score_condition(config, arm, lang, translator, stop, source_only=False):
    if source_only and arm=='base':
        return
    import numpy as np
    from ai_content_detector.rl_evasion.defender_comparison.defenders import DefenderPool
    from ai_content_detector.rl_evasion.defender_comparison.protocol import score_checkpointed, classification_metrics, validation_threshold
    directory=Path(config['directory'])
    rows=samples(config,arm,lang,translator=translator)
    data=read(directory/'partitions.json')['text']
    evaluation=data['test'][:sample_limit(config)]
    if [r['id'] for r in rows] != [r['id'] for r in evaluation]:
        raise ValueError('Incomplete or misaligned test condition')
    label=f'{arm}_{lang}' + (f'_{translator}' if translator else '')
    result={}
    humans=[dict(id=r['id'],text=r['human_text']) for r in evaluation]
    if not source_only:
        minimum_words=15 if any(s['family']=='diveye' for s in config['plan']['specifications'].values()) else 1
        scorable=[r for r in rows if len(r['text'].split())>=minimum_words]
        result['ensemble_scoring']=dict(n_generated=len(rows),n_scorable=len(scorable),minimum_words=minimum_words,
            unscorable_ids=[r['id'] for r in rows if len(r['text'].split())<minimum_words])
        pool=DefenderPool(config['plan']['specifications'],directory/'adapted_defenders',
            checkpoints=read(directory/'warmup_checkpoints.json'),retain_on_cpu=False)
        pool.inference_batch_sizes=config.get('detector_batch_sizes',{})
        try:
            scores=score_checkpointed(pool,[r['text'] for r in scorable],directory/'scores'/label)
            null=score_checkpointed(pool,[r['text'] for r in humans],directory/'scores'/'human')
        finally:
            pool.clear()
        ensemble=np.mean(list(scores.values()),axis=0)
        human=np.mean(list(null.values()),axis=0)
        calibration=read(directory/'validation_scores.json')
        n=len(next(iter(calibration.values())))//2
        threshold=validation_threshold(np.mean([v[n:] for v in calibration.values()],axis=0))
        result['ensemble_mean']=classification_metrics(ensemble,human)
        result['ensemble_validation_1pct_fpr']=classification_metrics(ensemble,human,threshold)
        result['diagnostic_members']={k:classification_metrics(v,null[k]) for k,v in scores.items()}
        valid=np.array([len(r['text'].split())>=20 for r in scorable],dtype=bool)
        result['valid_evasion_fraction_of_all_outputs']=float(np.sum(valid & (ensemble<.5))/len(rows))
        result['conditional_valid_evasion']=float(np.mean(ensemble[valid]<.5)) if valid.any() else None
        result['empty_outputs']=sum(not r['text'].strip() for r in rows)
        result['short_outputs_under_20_words']=sum(len(r['text'].split())<20 for r in rows)
    if arm != 'base':
        values=watermark_scores(config,arm,rows,label+'_watermark',stop)
        if source_only:
            save(directory/'scores'/f'{label}_source_watermark.json',values)
            return
        null=watermark_scores(config,arm,humans,arm+'_human_watermark',stop)
        if arm=='textseal2':
            threshold=3.
        else:
            calibration=[dict(id=r['id'],text=r['human_text']) for r in data['validation']]
            clean=samples(config,'base','en','validation')
            a=watermark_scores(config,arm,calibration,'latent_human_calibration',stop)
            b=watermark_scores(config,arm,clean,'latent_clean_calibration',stop)
            threshold=float(np.nextafter(max(np.quantile([r['score'] for r in a],.99,method='higher'),
                np.quantile([r['score'] for r in b],.99,method='higher')),np.inf))
        v=np.array([r['score'] for r in values]); h=np.array([r['score'] for r in null])
        result['watermark']=dict(n=len(v),threshold=threshold,detected=int(sum(v>=threshold)),
            detection_rate=float(np.mean(v>=threshold)),human_false_positive_rate=float(np.mean(h>=threshold)),
            eligible=sum(r['eligible'] for r in values))
        if translator:
            source=read(directory/f'{arm}_{lang}_watermark.json')['rows']
            positive=np.array([r['score']>=threshold for r in source])
            result['watermark']['source_detected']=int(sum(positive))
            result['watermark']['lost_given_source_detected']=float(np.mean(v[positive]<threshold)) if positive.any() else None
    save(directory/'scores'/f'{label}_metrics.json',result)


def quality(config, stop, arms=None):
    import torch
    import numpy as np
    import mauve
    import io
    import sqlite3
    from transformers import AutoModelForCausalLM, AutoTokenizer
    directory=Path(config['directory'])
    data=test_rows(config)
    model, tokenizer=None,None
    groups={'human':[dict(id=r['id'],text=r['human_text']) for r in data]}
    arms=tuple(arms or ('base','textseal2','latent'))
    if not arms or len(set(arms))!=len(arms) or set(arms)-{'base','textseal2','latent'}:
        raise ValueError('Unknown or repeated quality arm')
    groups.update({arm:samples(config,arm,'en') for arm in arms})
    features={}
    for name,rows in groups.items():
        cache=directory/'quality_features'/name
        cache.mkdir(parents=True,exist_ok=True)
        identity=dict(signature=config.get('quality_cache_signature',config['signature']),inputs=digest(rows),model=config['mauve']['model'])
        if (cache/'identity.json').exists() and read(cache/'identity.json')!=identity:
            raise ValueError('Quality feature cache inputs changed')
        save(cache/'identity.json',identity)
        connection=sqlite3.connect(cache/'blocks.sqlite3',isolation_level=None)
        connection.execute('CREATE TABLE IF NOT EXISTS blocks (start INTEGER PRIMARY KEY, payload BLOB NOT NULL)')
        blocks=[]
        for start in range(0,len(rows),8):
            cached=connection.execute('SELECT payload FROM blocks WHERE start=?',(start,)).fetchone()
            if cached is not None:
                blocks.append(np.load(io.BytesIO(cached[0]),allow_pickle=False))
                continue
            if model is None:
                tokenizer=AutoTokenizer.from_pretrained(config['mauve']['model'],local_files_only=True)
                tokenizer.pad_token=tokenizer.eos_token
                tokenizer.padding_side='left'
                model=AutoModelForCausalLM.from_pretrained(config['mauve']['model'],local_files_only=True,
                    dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
            batch=rows[start:start+8]
            if any(not r['text'].strip() for r in batch):
                raise ValueError('Empty texts require investigation before MAUVE')
            inputs=tokenizer([r['text'] for r in batch],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                vectors=model(**inputs,use_cache=False,output_hidden_states=True).hidden_states[-1][:,-1].float().cpu().numpy()
            payload=io.BytesIO()
            np.save(payload,vectors,allow_pickle=False)
            connection.execute('INSERT OR IGNORE INTO blocks VALUES (?,?)',(start,payload.getvalue()))
            committed=connection.execute('SELECT payload FROM blocks WHERE start=?',(start,)).fetchone()[0]
            blocks.append(np.load(io.BytesIO(committed),allow_pickle=False))
            if stop.requested:
                from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
                raise CheckpointPause('Quality feature block saved')
        connection.close()
        features[name]=np.concatenate(blocks)
    results={}
    for name in arms:
        results[name]={str(seed):float(mauve.compute_mauve(p_features=features['human'],q_features=features[name],
            num_buckets=config['mauve']['buckets'],seed=seed,verbose=False).mauve) for seed in config['mauve']['seeds']}
    merge_quality_results(directory,results,sample_limit(config))


def phase(config, stage, stop):
    import torch
    from ai_content_detector.watermarking import latent_text
    latent=latent_manifest(config)
    parts=stage.split(':')
    if stage=='preflight':
        from transformers import AutoTokenizer
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<70*2**30:
            raise ValueError('An allocated 80GB H100 is required')
        for translator in ('index','gemma'):
            tokenizer=AutoTokenizer.from_pretrained(config['models'][translator]['path'],local_files_only=True)
            for lang in ('zh-Hans','fr','cy'):
                tokenizer.apply_chat_template(translation_messages(translator,lang,'Test'),tokenize=False,add_generation_prompt=True)
        latent_text.preflight(latent)
    elif stage=='warmup': warmup(config,stop)
    elif stage=='teacher': latent_text.train(latent,stop)
    elif stage=='teacher_probe': latent_text.teacher_probe(latent,stop)
    elif stage=='student':
        if 'distillation_stop_step' in config:
            latent=dict(latent,distillation_stop_step=config['distillation_stop_step'])
        latent_text.fit(latent,stop)
    elif parts[0]=='generate': generate(config,*parts[1:],stop)
    elif parts[0]=='translate': translate(config,*parts[1:],stop)
    elif parts[0] in ('score','source_score'):
        score_condition(config,parts[1],parts[2],parts[3] if len(parts)>3 else None,stop,source_only=parts[0]=='source_score')
    elif parts[0]=='quality': quality(config,stop,parts[1:] or None)
    else: raise ValueError(stage)


def branch_stages(stages, branch):
    if branch not in ('distseal','non_distseal'):
        raise ValueError(branch)
    result=[]
    for stage in stages:
        parts=stage.split(':')
        if stage=='quality':
            result.append('quality:base:latent' if branch=='distseal' else 'quality:base:textseal2')
        elif branch=='distseal' and (stage in ('teacher','teacher_probe','student') or len(parts)>1 and parts[1]=='latent'):
            result.append(stage)
        elif branch=='non_distseal' and len(parts)>1 and parts[1] in ('base','textseal2'):
            result.append(stage)
    return result


def merge_quality_results(directory, results, n=5000):
    import fcntl
    directory=Path(directory)
    with (directory/'mauve.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=directory/'mauve.json'
        combined=read(path)['results'] if path.exists() else {}
        combined.update(results)
        save(path,dict(n=n,results=combined,
            protocol='Response-only Mistral final-token features, capped at 256 Mistral tokens, 25 buckets. Distributional proxy, not proof of preserved quality.'))


def report(config):
    import fcntl
    with (Path(config['directory'])/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _report(config)


def _report(config):
    directory=Path(config['directory'])
    lines=['# Qwen3.5-9B instruct translation results', '',
           'Primary AI detector: frozen warmed seven-member ensemble mean. Missing rows are pending.', '',
           '| Condition | N | Ensemble detection at 0.5 | Human FPR | Watermark detection |',
           '|---|---:|---:|---:|---:|']
    for path in sorted((directory/'scores').glob('*_metrics.json')):
        result=read(path)
        metric=result['ensemble_mean']
        watermark=result.get('watermark')
        rate=f"{watermark['detection_rate']:.2%}" if watermark else 'N/A'
        detection=f"{metric['ai_detection_rate']:.2%}" if metric['ai_detection_rate'] is not None else 'N/A'
        false_positive=f"{metric['human_false_positive_rate']:.2%}" if metric['human_false_positive_rate'] is not None else 'N/A'
        lines.append(f"| {path.stem.removesuffix('_metrics')} | {metric['num_ai']} | {detection} | {false_positive} | {rate} |")
    if (directory/'mauve.json').exists():
        lines += ['', f"## Direct-English MAUVE ({read(directory/'mauve.json')['n']:,} samples)", '',
                  '| Arm | Seed 42 | Seed 43 | Seed 44 |', '|---|---:|---:|---:|']
        for arm,values in read(directory/'mauve.json')['results'].items():
            lines.append('| '+arm+' | '+' | '.join(f"{values[str(k)]:.4f}" for k in (42,43,44))+' |')
        lines += ['', 'Custom response-only Mistral features. MAUVE alone does not establish preserved quality.']
    lines += ['', 'Raw detection is separate from language compliance and translation adequacy. See per-condition JSON for eligibility and validation-threshold results.', '']
    lines += ['Ensemble N counts scorable outputs. With DivEye, fewer than 15 words is unscorable; JSON records attempted counts and excluded IDs. Such outputs are not counted as successful valid evasions.', '']
    (directory/'results.md').write_text('\n'.join(lines))


def activate_source(config):
    source=Path(config['source']).resolve()
    for name,module in list(sys.modules.items()):
        if name=='ai_content_detector' or name.startswith('ai_content_detector.'):
            location=getattr(module,'__file__',None)
            if location and not Path(location).resolve().is_relative_to(source):
                raise ValueError('Live implementation imported before frozen source: '+name)
    sys.path[:0]=[str(source),str(source/'scripts')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    for name in ('warmup','cohort','selection','quality-template'):
        prep.add_argument('--'+name,type=Path,required=True)
    worker=sub.add_parser('phase')
    worker.add_argument('manifest',type=Path)
    worker.add_argument('stage')
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args)
        return
    config=read(args.manifest)
    directory=Path(config['directory'])
    if digest({k:v for k,v in config.items() if k!='signature'}) != config['signature']:
        raise ValueError('Manifest signature changed')
    for name,expected in config['frozen_hashes'].items():
        if sha(directory/name)!=expected:
            raise ValueError('Frozen source or input changed: '+name)
    activate_source(config)
    receipt=directory/'receipts'/(args.stage.replace(':','_')+'.json')
    if receipt.exists() and read(receipt).get('status')=='completed':
        saved=read(receipt)
        scoped=args.stage.startswith(('generate:','translate:','score:','source_score:','quality'))
        if not scoped or saved.get('num_samples')==sample_limit(config):
            return
    from ai_content_detector.rl_evasion.checkpointing import StopRequest,CheckpointPause
    from ai_content_detector.rl_evasion.config import seed_everything
    seed_everything(config['seed'])
    try:
        with StopRequest() as stop:
            phase(config,args.stage,stop)
            if stop.requested:
                raise CheckpointPause('Stage saved')
        save(receipt,dict(status='completed',stage=args.stage,num_samples=sample_limit(config)))
        report(config)
    except CheckpointPause:
        save(receipt,dict(status='paused',stage=args.stage))
        raise SystemExit(75)


if __name__=='__main__':
    main()
