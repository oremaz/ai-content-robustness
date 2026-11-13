"""Translation inputs and restart integrity without loading model weights."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('translation_driver', Path(__file__).resolve().parents[2]/'scripts/translation_experiments.py')
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_translategemma_uses_native_language_fields():
    for lang in ('zh-Hans', 'fr', 'cy'):
        message = driver.translation_messages('gemma', lang, 'source text')[0]
        assert message == {'role': 'user', 'content': [{'type': 'text', 'source_lang_code': lang,
                            'target_lang_code': 'en', 'text': 'source text'}]}
    with pytest.raises(ValueError):
        driver.translation_messages('gemma', 'en', 'text')


def test_index_prompt_keeps_source_as_data():
    message = driver.translation_messages('index', 'cy', 'test\nsource')[0]
    assert message['role'] == 'user'
    assert message['content'].endswith('\n\ntest\nsource')
    assert '威尔士语' in message['content'] and '英语' in message['content']


@pytest.mark.parametrize('key', ['id', 'human_text'])
def test_split_overlap_is_rejected(key):
    groups = {'train': [{'id':'a','human_text':'one'}], 'test': [{'id':'b','human_text':'two'}]}
    driver.check_cohorts(groups)
    groups['test'][0][key] = groups['train'][0][key]
    with pytest.raises(ValueError, match='overlap'):
        driver.check_cohorts(groups)


def test_restart_rejects_changed_order_and_protocol(tmp_path):
    config = {'directory':str(tmp_path), 'signature':'frozen'}
    expected = [{'id':'a'}, {'id':'b'}]
    path, rows = driver.rows_file(config, 'samples', expected)
    driver.retained(config, path, expected[:1], SimpleNamespace(requested=False))
    assert driver.rows_file(config, 'samples', expected)[1] == expected[:1]
    with pytest.raises(ValueError):
        driver.rows_file(config, 'samples', expected[::-1])
    with pytest.raises(ValueError):
        driver.rows_file(dict(config, signature='different'), 'samples', expected)


def test_restart_accepts_only_declared_predecessor_signature(tmp_path):
    expected = [{"id": "a"}]
    driver.save(tmp_path/"samples.json", {"signature": "original", "rows": expected})
    config = {"directory": str(tmp_path), "signature": "revised", "compatible_row_signatures": ["original"]}
    assert driver.rows_file(config, "samples", expected)[1] == expected
    with pytest.raises(ValueError):
        driver.rows_file(dict(config, compatible_row_signatures=[]), "samples", expected)


def test_frozen_source_precedes_live_checkout(tmp_path, monkeypatch):
    import sys

    source = tmp_path / 'frozen'
    monkeypatch.syspath_prepend(str(tmp_path / 'live'))
    monkeypatch.setitem(sys.modules, 'ai_content_detector', SimpleNamespace(__file__=str(tmp_path/'live/__init__.py')))
    with pytest.raises(ValueError, match='Live implementation'):
        driver.activate_source({'source': str(source)})
    for name in list(sys.modules):
        if name == 'ai_content_detector' or name.startswith('ai_content_detector.'):
            monkeypatch.delitem(sys.modules, name)
    driver.activate_source({'source': str(source)})
    assert sys.path[:2] == [str(source), str(source/'scripts')]


def test_latent_length_filter_keeps_test_cohort_and_original_rows():
    def row(name, length):
        return {'id':name,'prompt_length':5,'tokens':list(range(5+length))}
    data={'text':{'train':[row('short',127),row('full',128),row('long',200)],
                  'validation':[row('vshort',10),row('vfull',128),row('vlong',150)],
                  'test':[row('heldout',23)]}}
    filtered=driver.latent_training_partitions(data,128,2)
    assert [r['id'] for r in filtered['text']['train']]==['full','long']
    assert [r['id'] for r in filtered['text']['validation']]==['vfull','vlong']
    assert filtered['text']['test'] is data['text']['test']
    assert len(data['text']['train'])==3
    with pytest.raises(ValueError,match='Insufficient'):
        driver.latent_training_partitions(data,201,2)


def test_parallel_branches_are_disjoint_and_cover_experiment():
    stages=['preflight','warmup','teacher','teacher_probe','student','generate:base:en:test',
            'generate:textseal2:en:test','score:textseal2:en','generate:latent:en:test','score:latent:en',
            'quality','generate:base:cy:test','translate:base:cy:index','score:base:cy:index',
            'generate:textseal2:cy:test','translate:textseal2:cy:gemma','score:textseal2:cy:gemma',
            'generate:latent:cy:test','translate:latent:cy:gemma','score:latent:cy:gemma']
    distseal=driver.branch_stages(stages,'distseal')
    other=driver.branch_stages(stages,'non_distseal')
    assert not set(distseal)&set(other)
    assert 'student' in distseal and 'student' not in other
    assert 'quality:base:latent' in distseal
    assert 'quality:base:textseal2' in other
    assert (set(distseal)|set(other))-{'quality:base:latent','quality:base:textseal2'} == set(stages)-{'preflight','warmup','quality'}
    with pytest.raises(ValueError):
        driver.branch_stages(stages,'unknown')


def test_parallel_quality_results_merge_without_losing_arms(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def submit(arm):
        for _ in range(5):
            driver.merge_quality_results(tmp_path, {'base': {'42': .8}, arm: {'42': .7}})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(submit,arm) for arm in ('latent','textseal2')]
        for future in futures:
            future.result()
    assert set(driver.read(tmp_path/'mauve.json')['results'])=={'base','latent','textseal2'}


def test_clean_source_watermark_is_not_applicable():
    assert driver.score_condition({}, 'base', 'cy', None, None, source_only=True) is None


def test_pilot_view_keeps_extra_saved_texts_for_expansion(tmp_path):
    rows=[{'id':str(i),'text':'saved '+str(i)} for i in range(5000)]
    driver.save(tmp_path/'base_en_test.json',{'signature':'original','rows':rows})
    config={'directory':str(tmp_path),'signature':'pilot','compatible_row_signatures':['original'],'num_samples':2000}
    assert len(driver.samples(config,'base','en'))==2000
    assert len(driver.rows_file(config,'base_en_test',rows)[1])==5000
    assert len(driver.samples(dict(config,num_samples=5000),'base','en'))==5000
    for invalid in (0,5001,True):
        with pytest.raises(ValueError):
            driver.sample_limit(dict(config,num_samples=invalid))


def test_pilot_mauve_reports_actual_sample_count(tmp_path):
    driver.merge_quality_results(tmp_path,{'base':{'42':.2,'43':.2,'44':.2}},n=2000)
    assert driver.read(tmp_path/'mauve.json')['n']==2000
    driver.report({'directory':str(tmp_path)})
    assert '2,000 samples' in (tmp_path/'results.md').read_text()


def test_unscorable_output_is_retained_and_not_a_successful_evasion(tmp_path, monkeypatch):
    import sys

    rows=[{'id':name,'text':' '.join(['word']*count)} for name,count in [('short',12),('medium',18),('valid',25)]]
    driver.save(tmp_path/'base_en_test.json',{'rows':rows})
    driver.save(tmp_path/'partitions.json',{'text':{'test':[{'id':r['id'],'human_text':'human '*25} for r in rows]}})
    driver.save(tmp_path/'warmup_checkpoints.json',{})
    driver.save(tmp_path/'validation_scores.json',{'diveye':[.9,.1]})

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def clear(self):
            pass

    calls=[]
    def score(pool, texts, path):
        calls.append(texts)
        assert all(len(text.split())>=15 for text in texts)
        return {'diveye':[.2]*len(texts)}

    def metrics(ai, human, threshold=.5):
        return {'num_ai':len(ai),'num_human':len(human)}

    prefix='ai_content_detector.rl_evasion.defender_comparison.'
    monkeypatch.setitem(sys.modules,prefix+'defenders',SimpleNamespace(DefenderPool=Pool))
    monkeypatch.setitem(sys.modules,prefix+'protocol',SimpleNamespace(score_checkpointed=score,
        classification_metrics=metrics,validation_threshold=lambda values:.4))
    config={'directory':str(tmp_path),'num_samples':3,'plan':{'specifications':{'diveye':{'family':'diveye'}}}}
    driver.score_condition(config,'base','en',None,None)
    result=driver.read(tmp_path/'scores/base_en_metrics.json')
    assert result['ensemble_mean']=={'num_ai':2,'num_human':3}
    assert result['ensemble_scoring']['unscorable_ids']==['short']
    assert result['ensemble_scoring']['n_generated']==3
    assert result['valid_evasion_fraction_of_all_outputs']==pytest.approx(1/3)
    assert result['conditional_valid_evasion']==1.
    assert driver.read(tmp_path/'base_en_test.json')['rows']==rows
