import pytest

from ai_content_detector.watermarking.preservation import select_preserving_candidate


def test_blind_quality_selection_and_no_eligible_attack():
    rows = [dict(name='strong', step=500, anchor_weight=0., preserved=27, n=32, detector_score=0.),
            dict(name='regularized', step=500, anchor_weight=4., preserved=28, n=32, detector_score=1.),
            dict(name='very_regularized', step=500, anchor_weight=16., preserved=32, n=32, detector_score=0.),
            dict(name='short', step=100, anchor_weight=0., preserved=32, n=32, detector_score=0.)]
    assert select_preserving_candidate(rows, 28)['name'] == 'regularized'
    for row in rows:row['detector_score'] = 1-row['detector_score']
    assert select_preserving_candidate(rows, 28)['name'] == 'regularized'
    assert select_preserving_candidate(rows[:1], 28) is None


def test_selection_rejects_invalid_quality_evidence():
    row = dict(name='candidate', step=100, anchor_weight=1., preserved=33, n=32)
    with pytest.raises(ValueError):select_preserving_candidate([row], 28)
    row['preserved'] = 30
    with pytest.raises(ValueError):select_preserving_candidate([row, row], 28)
