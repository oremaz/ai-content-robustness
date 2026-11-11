"""Matched-image pairing, integrity and quality metric contracts."""

from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from ai_content_detector.watermarking.image_quality import (
    ImageQuality, align_by_index, paired_psnr, read_pair,
)
from ai_content_detector.watermarking.closed_image import file_hash


def test_alignment_preserves_indices_and_class_order():
    reference = [{"index": 1, "label": 5}, {"index": 2, "label": 7}]
    assert align_by_index(reference, reference[::-1]) == reference
    with pytest.raises(ValueError, match="indices"):
        align_by_index(reference, reference[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        align_by_index(reference, reference+reference[:1])
    with pytest.raises(ValueError, match="class labels"):
        align_by_index(reference, [{"index": 1, "label": 7}, {"index": 2, "label": 5}])


def test_pair_reader_checks_content_and_native_geometry(tmp_path):
    first, second = tmp_path/"a.png", tmp_path/"b.png"
    Image.new("RGB", (32, 32), "red").save(first)
    Image.new("RGB", (32, 32), "blue").save(second)
    row = {"index": 4, "reference": str(first), "candidate": str(second),
           "reference_sha256": file_hash(first), "candidate_sha256": file_hash(second)}
    assert [i.size for i in read_pair(row)] == [(32, 32), (32, 32)]
    Image.new("RGB", (64, 64), "blue").save(second)
    with pytest.raises(ValueError, match="PNG changed"):
        read_pair(row)
    row["candidate_sha256"] = file_hash(second)
    with pytest.raises(ValueError, match="native resolutions"):
        read_pair(row)


def test_psnr_uses_unit_pixel_range_and_explicit_identity_ceiling():
    first = torch.zeros(2, 3, 16, 16)
    second = first.clone();second[1] = .1
    torch.testing.assert_close(paired_psnr(first, second), torch.tensor([120., 20.]))
    with pytest.raises(ValueError, match="matching"):
        paired_psnr(first, second[:1])


def test_image_and_class_clip_use_the_corresponding_pair_and_label():
    engine = ImageQuality.__new__(ImageQuality)
    engine.device = "cpu"
    engine.categories = ["first", "second"]
    engine.text_features = torch.eye(2)
    engine.lpips = lambda a,b: torch.zeros(len(a), 1, 1, 1)
    features = torch.tensor([[1.,0.], [0.,1.], [1.,0.], [1.,1.]])
    engine.clip = SimpleNamespace(get_image_features=lambda **kwargs: SimpleNamespace(pooler_output=features))
    class Batch(dict):
        def to(self, device):
            return self
    engine.processor = lambda **kwargs: Batch(pixel_values=torch.empty(4, 3, 32, 32))
    images = [Image.new("RGB", (32, 32), "red"), Image.new("RGB", (32, 32), "blue")]
    scores = engine.score(images, images, [0, 1])
    assert scores[0]["clip_alignment"] == pytest.approx(1.)
    assert scores[1]["clip_alignment"] == pytest.approx(2**-.5)
    assert scores[1]["clip_image_cosine"] == pytest.approx(2**-.5)
    assert all(r["reference_clip_alignment"] == pytest.approx(1.) for r in scores)
    assert all(r["ssim"] == pytest.approx(1.) and r["psnr"] == pytest.approx(120.) for r in scores)
    with pytest.raises(ValueError, match="canonical"):
        engine.score(images, images, [0, 2])
