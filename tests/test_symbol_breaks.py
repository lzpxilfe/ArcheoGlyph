from archeoglyph.symbol_breaks import (
    MODE_EQUAL_INTERVAL,
    MODE_NATURAL_BREAKS,
    MODE_QUANTILE,
    compute_breaks,
    jenks_breaks,
)


def test_equal_interval_is_evenly_spaced():
    breaks = compute_breaks(range(0, 11), 5, MODE_EQUAL_INTERVAL)
    assert breaks == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_quantile_breaks_cover_range_and_increase():
    values = [1, 2, 2, 3, 10, 20, 30, 40, 100]
    breaks = compute_breaks(values, 4, MODE_QUANTILE)
    assert breaks[0] == 1.0 and breaks[-1] == 100.0
    assert all(b < a for b, a in zip(breaks, breaks[1:]))


def test_jenks_separates_two_clusters():
    values = [1, 2, 3, 100, 101, 102]
    breaks = jenks_breaks(sorted(values), 2)
    assert breaks[0] == 1.0 and breaks[-1] == 102.0
    assert 3.0 <= breaks[1] <= 100.0


def test_compute_breaks_handles_degenerate_input():
    assert compute_breaks([], 5, MODE_NATURAL_BREAKS) == []
    assert compute_breaks([7, 7, 7], 3, MODE_NATURAL_BREAKS) == [7.0, 8.0]
    assert compute_breaks([5], 3, MODE_QUANTILE) == [5.0, 6.0]
