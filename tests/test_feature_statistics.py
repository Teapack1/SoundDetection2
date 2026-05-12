"""Train-only feature-standardization sampling and moment regressions."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest


def _row(class_name: str, group: str, index: int) -> dict[str, str]:
    return {
        "path": f"/audio/{class_name}/{group}/{index}.wav",
        "class_name": class_name,
        "source_split_group": group,
        "origin": "recorded",
    }


def test_feature_stats_selection_is_seeded_and_balances_classes_and_groups():
    from sed.data.feature_stats import balanced_stats_rows
    from sed.data.splits import split_group_key

    rows = (
        [_row("_negative", "large", i) for i in range(100)]
        + [_row("_negative", "small", i) for i in range(4)]
        + [_row("glass", "g1", i) for i in range(8)]
        + [_row("glass", "g2", i) for i in range(8)]
    )
    selected = balanced_stats_rows(rows, limit=16, seed=43)
    again = balanced_stats_rows(rows, limit=16, seed=43)

    assert [row["path"] for row in selected] == [row["path"] for row in again]
    assert Counter(row["class_name"] for row in selected) == {
        "_negative": 8,
        "glass": 8,
    }
    by_class_group = Counter(
        (row["class_name"], split_group_key(row)) for row in selected
    )
    assert by_class_group[("_negative", "large")] == 4
    assert by_class_group[("_negative", "small")] == 4
    assert by_class_group[("glass", "g1")] == 4
    assert by_class_group[("glass", "g2")] == 4


def test_feature_stats_welford_batch_merge_matches_numpy_population_moments():
    from sed.data.feature_stats import merge_value_moments

    rng = np.random.default_rng(7)
    batches = [
        rng.normal(-2.0, 0.5, size=(3, 17, 11)),
        rng.normal(3.0, 2.0, size=(5, 17, 11)),
        rng.normal(0.0, 1.0, size=(1, 17, 11)),
    ]
    count, mean, m2 = 0, 0.0, 0.0
    for batch in batches:
        count, mean, m2 = merge_value_moments(count, mean, m2, batch)

    expected = np.concatenate([batch.reshape(-1) for batch in batches])
    assert count == expected.size
    assert mean == pytest.approx(np.mean(expected, dtype=np.float64), abs=1e-12)
    assert np.sqrt(m2 / count) == pytest.approx(
        np.std(expected, dtype=np.float64), abs=1e-12,
    )


def test_per_band_welford_matches_numpy_per_band_moments():
    from sed.data.feature_stats import _merge_band_moments

    rng = np.random.default_rng(11)
    batches = [
        rng.normal(-2.0, 0.5, size=(3, 17, 11)),
        rng.normal(3.0, 2.0, size=(5, 17, 11)),
        rng.normal(0.0, 1.0, size=(1, 17, 11)),
    ]
    count, mean, m2 = 0, None, None
    for batch in batches:
        count, mean, m2 = _merge_band_moments(count, mean, m2, batch)

    stacked = np.concatenate(
        [np.transpose(b, (1, 0, 2)).reshape(17, -1) for b in batches], axis=1
    )
    assert count == stacked.shape[1]
    np.testing.assert_allclose(mean, stacked.mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(np.sqrt(m2 / count), stacked.std(axis=1), atol=1e-12)
