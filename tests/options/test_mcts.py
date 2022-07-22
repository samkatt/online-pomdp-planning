"""Tests :mod:`online_pomdp_planning.options.mcts`"""

import random
from functools import partial

import pytest

from online_pomdp_planning.options.mcts import Option, apply_option
from online_pomdp_planning.types import Action


def is_observation_stop_condition(_, o, stop_o) -> bool:
    """A :class:`StopCondition` that returns `True` if `o` equals `stop_o`"""
    return o == stop_o


def macro_action_policy(s, o, a) -> Action:
    """This option :class:`Policy` always returns `a`"""
    return a


def test_apply_option_stop_condition():
    def random_observation_simulator(s, a):
        """A simulator which randomly returns 'stop o'"""
        return (
            None,
            random.choice(["some o", "some other o", "stop o", True, 10]),
            1,
            False,
        )

    def random_terminal_simulator(s, a):
        """A simulator which randomly terminates"""
        return (
            None,
            random.choice(["some o", "some other o", True, 10]),
            1,
            random.random() > 0.5,
        )

    option = Option(
        lambda o: None,
        partial(is_observation_stop_condition, stop_o="stop o"),
    )

    # test that stopping condition based on observation works
    s, o, rewards, t = apply_option(option, 0, "some o", random_observation_simulator)
    assert s is None
    assert o == "stop o"
    assert len(rewards) > 0 and sum(rewards) > 1
    assert not t

    # test that terminating simulator works
    s, o, __, t = apply_option(option, 0, "some o", random_terminal_simulator)
    assert s is None
    assert o != "stop o"
    assert len(rewards) > 0 and sum(rewards) > 1
    assert t


if __name__ == "__main__":
    pytest.main([__file__])
