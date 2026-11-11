"""Portable result links, fixed sampling and faithful qualitative rendering."""

import html
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from watermark_report_assets import full_text, image_grid, sample_positions, sha, text_grid, validate_publication
from publish_watermark_results import binomial_ci
import watermark_report_assets as assets


def test_sampling_is_fixed_evenly_spaced_and_never_filters_failures():
    assert sample_positions(512)==[0,64,128,192,256,320,384,448]
    assert sample_positions(256)==[0,32,64,96,128,160,192,224]
    assert sample_positions(330)==[0,41,82,123,165,206,247,288]
    with pytest.raises(ValueError):sample_positions(7)


def test_full_verbatim_text_is_preserved_and_html_is_escaped():
    original='A <script>alert("x")</script>\n```a```\n'+chr(8212)+" end"
    rendered=full_text(original)
    assert "<script>" not in rendered and "&#8212;" in rendered
    assert html.unescape(rendered[len('<pre>'):-len('</pre>')])==original


def test_image_grid_contains_all_sixteen_correct_panels(tmp_path):
    pairs=[]
    for i in range(8):
        pairs.append({"index":i,"before":Image.new('RGB',(16,16),(20*i,30,40)),
                      "after":Image.new('RGB',(16,16),(20*i,60,80)),"before_label":"before","after_label":"after"})
    path=tmp_path/'grid.webp';image_grid(path,'Test',pairs,panel_size=64)
    with Image.open(path) as grid:
        for i in range(8):
            x=12+(i%2)*2*76;y=90+(i//2)*151+75
            assert grid.getpixel((x+32,y+32))==(20*i,30,40)
            assert grid.getpixel((x+76+32,y+32))==(20*i,60,80)
    with pytest.raises(ValueError):image_grid(path,'Test',pairs[:7])
    with pytest.raises(ValueError):text_grid(path,'Test',[])


def test_portable_audit_checks_hashes_links_and_pair_coverage(tmp_path):
    (tmp_path/'assets').mkdir()
    asset=tmp_path/'assets/grid.webp';Image.new('RGB',(8,8),'red').save(asset,'WEBP',lossless=True)
    page=tmp_path/'README.md';page.write_text('[Grid](assets/grid.webp)\n<pre>[Literal example](not-a-link)</pre>\n')
    def manifest():
        (tmp_path/'publication.json').write_text(json.dumps({'files':{'README.md':sha(page),'assets/grid.webp':sha(asset)},
          'galleries':[{'sample_ids':list(range(8)),'panels':16,'asset':'assets/grid.webp'}]}))
    manifest();assert validate_publication(tmp_path)['pairs_per_gallery']==8
    page.write_text('[Missing](missing.md)');manifest()
    with pytest.raises(ValueError,match='Broken'):validate_publication(tmp_path)
    page.write_text('[Local only](../results/run.json)');manifest()
    with pytest.raises(ValueError,match='Nonportable'):validate_publication(tmp_path)
    page.write_text('[Absolute](/etc/passwd)');manifest()
    with pytest.raises(ValueError,match='Nonportable'):validate_publication(tmp_path)
    page.write_text('[Grid](assets/grid.webp)');manifest();asset.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):validate_publication(tmp_path)


def test_binomial_interval_treats_zero_as_uncertain():
    low,high=binomial_ci(0,256)
    assert low==0 and .01<high<.02
    assert binomial_ci(256,256)[1]==1


def test_unicode_preview_is_explicit_without_changing_full_text(tmp_path,monkeypatch):
    original='A box '+chr(0x1f4e6)
    pairs=[{'before':original,'after':original,'before_label':'before','after_label':'after'} for _ in range(8)]
    observed=[]
    original_wrap=assets.wrapped
    def capture(draw,text,face,width):
        observed.append(text)
        return original_wrap(draw,text,face,width)
    monkeypatch.setattr(assets,'wrapped',capture)
    text_grid(tmp_path/'text.webp','Text',pairs)
    assert any('[U+1F4E6]' in text for text in observed)
    assert all(pair['before']==original for pair in pairs)
