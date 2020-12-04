"""Defines some basic environments to test planners in for integration tests"""
import random
from typing import List, Tuple

from online_pomdp_planning.mcts import create_POUCT
from online_pomdp_planning.types import Action, Observation, State


class Tiger:
    """The Tiger POMDP environment"""

    L = 0
    R = 1
    H = 2

    H_REWARD = -1
    OPEN_CORRECT_REWARD = 10
    OPEN_INCORRECT_REWARD = -100

    @staticmethod
    def actions() -> List[Action]:
        """returns actions in the tiger problem """
        return [Tiger.L, Tiger.R, Tiger.H]

    @staticmethod
    def sample_observation(s: State) -> Observation:
        """85% hear tiger correctly"""

    @staticmethod
    def get_reward(s: State, a: Action) -> float:
        """TODO"""

    @staticmethod
    def sim(s: State, a: Action) -> Tuple[State, Observation, float, bool]:
        """Simulates the tiger dynamics"""

        if a == Tiger.H:
            o = Tiger.sample_observation(s)
            return (s, o, Tiger.H_REWARD, False)

        o = random.choice([Tiger.L, Tiger.R])
        r = Tiger.OPEN_CORRECT_REWARD if s == a else Tiger.OPEN_INCORRECT_REWARD

        s = random.choice([Tiger.L, Tiger.R])

        return s, o, r, True


def uniform_tiger_belief():
    """Sampling returns 'left' and 'right' state equally"""
    return random.choice([Tiger.L, Tiger.R])


def tiger_left_belief():
    """Sampling returns 'left' state"""
    return Tiger.L


def tiger_right_belief():
    """Sampling returns 'right' state"""
    return Tiger.R


def test_pouct():
    """tests :py:func:`~online_pomdp_planning.mcts.create_POUCT` on Tiger"""

    planner = create_POUCT(Tiger.actions(), Tiger.sim, ucb_constant=100)

    assert planner(uniform_tiger_belief, 16384) == Tiger.H
    assert planner(tiger_left_belief, 16384) == Tiger.L
    assert planner(tiger_right_belief, 16384) == Tiger.R
