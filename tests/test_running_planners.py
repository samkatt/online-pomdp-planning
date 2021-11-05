"""Defines some basic environments to test planners in for integration tests"""
import random
from collections import Counter
from typing import Counter as CounterType
from typing import Dict, List, Mapping, Tuple

import pytest

from online_pomdp_planning.mcts import (
    MuzeroInferenceOutput,
    create_muzero,
    create_POUCT,
    create_POUCT_with_state_models,
)
from online_pomdp_planning.types import Action


class Tiger:
    """The Tiger POMDP environment"""

    L = 0
    R = 1
    H = 2

    H_REWARD = -1
    OPEN_CORRECT_REWARD = 10
    OPEN_INCORRECT_REWARD = -100

    @staticmethod
    def actions() -> List[int]:
        """returns actions in the tiger problem"""
        return [Tiger.L, Tiger.R, Tiger.H]

    @staticmethod
    def sample_observation(s: int) -> int:
        """85% hear tiger correctly"""
        if random.uniform(0, 1) < 0.85:
            return s
        return int(not s)

    @staticmethod
    def sim(s: int, a: int) -> Tuple[int, int, float, bool]:
        """Simulates the tiger dynamics"""

        if a == Tiger.H:
            o = Tiger.sample_observation(s)
            return (s, o, Tiger.H_REWARD, False)

        o = random.choice([Tiger.L, Tiger.R])
        r = Tiger.OPEN_CORRECT_REWARD if s == a else Tiger.OPEN_INCORRECT_REWARD

        s = random.choice([Tiger.L, Tiger.R])

        return s, o, r, True

    @staticmethod
    def state_evaluation(s) -> Tuple[float, Mapping[int, float]]:
        """A 'state-based model' for the Tiger

        Hard-coded evaluation and prior for this problem.
        """
        good_door = s
        bad_door = int(not s)
        return 4.0, {Tiger.H: 5.0, good_door: 4.0, bad_door: -20}


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
    """tests :func:`~online_pomdp_planning.mcts.create_POUCT` on Tiger"""

    planner = create_POUCT(Tiger.actions(), Tiger.sim, 2 * 16384, ucb_constant=100)

    action, info = planner(uniform_tiger_belief)
    assert action == Tiger.H
    assert info["iteration"] == 16384 * 2

    planner = create_POUCT(Tiger.actions(), Tiger.sim, 16384, ucb_constant=100)
    action, info = planner(tiger_left_belief)
    assert action == Tiger.L
    assert info["iteration"] == 16384

    planner = create_POUCT(
        Tiger.actions(),
        Tiger.sim,
        16384,
        ucb_constant=100,
        leaf_eval=lambda s, o, t, info: 0,
    )

    action, info = planner(tiger_right_belief)
    assert action == Tiger.R


def test_pouct_with_prior():
    """tests :func:`~online_pomdp_planning.mcts.create_POUCT_with_state_models` on Tiger"""

    planner = create_POUCT_with_state_models(
        Tiger.actions(), Tiger.sim, 2 * 16384, Tiger.state_evaluation, ucb_constant=100
    )

    action, info = planner(uniform_tiger_belief)
    assert action == Tiger.H
    assert info["iteration"] == 16384 * 2

    planner = create_POUCT(Tiger.actions(), Tiger.sim, 16384, ucb_constant=100)
    action, info = planner(tiger_left_belief)
    assert action == Tiger.L
    assert info["iteration"] == 16384

    planner = create_POUCT(
        Tiger.actions(),
        Tiger.sim,
        16384,
        ucb_constant=100,
        leaf_eval=lambda s, o, t, info: 0,
    )

    action, info = planner(tiger_right_belief)
    assert action == Tiger.R


def test_muzero():
    """tests :func:`create_muzero`"""

    num_sims = 128
    c1, c2 = 0.75, 20
    gamma = 0.9
    noise_alpha = 10
    noise_factor = 0.2

    # `history`: [a0 o0 a1 ... oT]
    # `latent_state`: {"history": `history`, "actions": [aT+1 aT+2 ...]}"

    def value_of(actions) -> float:
        """Value of a 'latent' state of having taken ``actions``

        If opened door in ``actions`` => 0, else length of ``actions`` as proxy
        """
        if not all(a == Tiger.H for a in actions):
            return 0
        return 2 * len(actions)

    def how_bad_is_action(history, a) -> int:
        """Score from 0 ... 4 of how about ``a`` is given ``history``"""
        observation_counts: CounterType[int] = Counter({Tiger.L: 0, Tiger.R: 0})
        observation_counts.update(history[1::2])

        obs_diff = min(
            abs(observation_counts[Tiger.L] - observation_counts[Tiger.R]), 2
        )
        most_observed = observation_counts.most_common(1)[0][0]

        return obs_diff + (2 * (a == most_observed))

    def reward_of(history, actions, a) -> float:
        """reward of taking ``a`` after simulating doing ``actions`` after ``history``

        - if opened door in ``actions`` then zero
        - else when listening then -1
        - else some magic that rewards opening door seen in history
        """
        # opened door means nothing!!
        if not all(a == Tiger.H for a in actions):
            return 0

        # listening is always -1
        if a == Tiger.H:
            return -1

        # doing a lot of actions at some point just returns 0
        if len(actions) > 4:
            return 0.0

        return [-100, -75, -45, 2, 5][how_bad_is_action(history, a)]

    def policy_of(history) -> Dict[Action, float]:
        return {a: 1 / 3 for a in [Tiger.H, Tiger.L, Tiger.R]}

    def init_inference(history) -> MuzeroInferenceOutput:
        """list of observations => initial inference"""
        return MuzeroInferenceOutput(
            # the longer in the episode we are, the higher the value
            value=value_of([]),
            reward=0,
            policy=policy_of(history),
            latent_state={"history": history, "actions": []},
        )

    def rec_inference(state, action) -> MuzeroInferenceOutput:
        """list of observations and action => inference"""
        new_s = {"history": state["history"], "actions": state["actions"] + [action]}

        return MuzeroInferenceOutput(
            value=value_of(new_s["actions"]),
            reward=reward_of(state["history"], state["actions"], action),
            policy=policy_of(state["history"]),
            latent_state=new_s,
        )

    planner = create_muzero(
        init_inference,
        rec_inference,
        num_sims,
        c1,
        c2,
        gamma,
        noise_alpha,
        noise_factor,
    )

    action, info = planner([Tiger.H, Tiger.L, Tiger.H, Tiger.L])
    assert "q_statistic" in info
    assert info["iteration"] == num_sims
    assert action == Tiger.L

    action, info = planner([Tiger.H, Tiger.R])
    assert "q_statistic" in info
    assert info["iteration"] == num_sims
    assert action == Tiger.R

    action, info = planner([])
    assert "q_statistic" in info
    assert info["iteration"] == num_sims
    assert action == Tiger.H

    action, info = planner([Tiger.H, Tiger.L, Tiger.H, Tiger.R])
    assert "q_statistic" in info
    assert info["iteration"] == num_sims
    assert action == Tiger.H


if __name__ == "__main__":
    pytest.main([__file__])
