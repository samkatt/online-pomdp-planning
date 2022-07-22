"""Defines some basic environments to test planners in for integration tests"""
import random
from functools import partial

import pytest

from online_pomdp_planning.options.mcts import (
    Option,
    action_to_option,
    create_POUCT_with_options,
)
from tests.test_running_planners import (
    Tiger,
    tiger_left_belief,
    tiger_right_belief,
    uniform_tiger_belief,
)


def random_stop_condition(h, p) -> bool:
    """Stop condition that randomly terminates (with probability `p`)"""
    return random.random() < p


def always_stop_condition(h) -> bool:
    """A :class:`StopCondition` that always terminates"""
    return True


def test_options_pouct():
    """tests :func:`~online_pomdp_planning.options.mcts.create_POUCT_with_options` on Tiger"""

    large_num_sum = 12345
    listen_option = Option(lambda h: Tiger.H, partial(random_stop_condition, p=0.3))
    open_left_option = action_to_option(Tiger.L, always_stop_condition)
    open_right_option = action_to_option(Tiger.R, always_stop_condition)

    options = [
        listen_option,
        open_left_option,
        open_right_option,
    ]
    planner = create_POUCT_with_options(
        options, Tiger.sim, large_num_sum, ucb_constant=1000
    )

    action, info = planner(uniform_tiger_belief)
    assert action == listen_option
    assert sum(stat["n"] for stat in info["tree_root_stats"].values()) == large_num_sum
    assert info["iteration"] == large_num_sum

    action, info = planner(tiger_left_belief)
    assert sum(stat["n"] for stat in info["tree_root_stats"].values()) == large_num_sum
    assert action == open_left_option
    assert info["iteration"] == large_num_sum

    action, info = planner(tiger_right_belief)
    assert sum(stat["n"] for stat in info["tree_root_stats"].values()) == large_num_sum
    assert action == open_right_option
    assert info["iteration"] == large_num_sum


if __name__ == "__main__":
    pytest.main([__file__])
