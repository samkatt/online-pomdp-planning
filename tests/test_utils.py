"""Tests `online_pomdp_planning.utils`"""

import pytest

from online_pomdp_planning.utils import MovingStatistic


@pytest.mark.parametrize(
    "vals,mean,minimum,maximum",
    [
        ([], 0, float("Inf"), float("-Inf")),
        ([0], 0, 0, 0),
        ([0, 1], 0.5, 0, 1),
        ([-1, 0], -0.5, -1, 0),
        ([-1, 10, 5.5, 100.5], 28.75, -1, 100.5),
    ],
)
def test_moving_statistic(vals, mean, minimum, maximum):
    """Tests :class:`online_pomdp_planning.utils.MovingStatistic`"""
    s = MovingStatistic()

    for v in vals:
        s.add(v)

    assert s.mean == mean
    assert s.min == minimum
    assert s.max == maximum
    assert s.num == len(vals)
