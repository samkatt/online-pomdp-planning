"""Some test on learning state-based values"""
import random
from collections import Counter
from functools import partial
from typing import Callable, List

import numpy as np

left = 0
right = 1

states = [left, right]
observations = [left, right]

Belief = Callable[[], int]


def uniform() -> int:
    """uniform tiger belief"""
    return random.choices(states, weights=[0.5, 0.5])[0]


def listen_once(left_or_right: int) -> int:
    """belief after listening once"""
    weights = [0.85, 0.15] if left_or_right == left else [0.15, 0.85]
    return random.choices(states, weights=weights)[0]


def listen_twice(left_or_right: int) -> int:
    """belief after listening twice"""
    weights = [0.96, 0.04] if left_or_right == left else [0.04, 0.96]
    return random.choices(states, weights=weights)[0]


def original_step(v_b: List[float], p_b: List[Belief], v_s: List[float]):
    """Updates ``v_s`` by applying a single ``gradient_update``"""

    # sample belief and state
    sample_belief = random.randint(0, len(v_b) - 1)
    b, v = p_b[sample_belief], v_b[sample_belief]
    s = b()

    # gradient MSE update step
    v_s[s] = v_s[s] + 0.001 * (v - v_s[s])


def bias_step(v_b: List[float], p_b: List[Belief], v_s: List[float]):
    """Updates ``v_s`` by applying a single ``gradient_update`` on bias target only"""

    # sample belief and states
    sample_belief = random.randint(0, len(v_b) - 1)
    b, v = p_b[sample_belief], v_b[sample_belief]
    sample_states = [b() for _ in range(100)]

    # estimate expected value
    exp_val = np.mean([v_s[s] for s in sample_states])

    # update step according to bias gradient
    for s, p in Counter(sample_states).items():
        v_s[s] = v_s[s] + (0.00001 * p) * (v - exp_val)


def get_estimated_belief_value(b: Belief, v_s: List[float]) -> float:
    """(sample) estimate of value of ``b`` according to ``v_s``"""
    return np.mean([v_s[b()] for _ in range(100000)])


if __name__ == "__main__":

    belief_distr = [
        uniform,
        partial(listen_once, left),
        partial(listen_twice, left),
        partial(listen_once, right),
        # partial(listen_once, heard_left=False),
    ]
    belief_values = [3.0, 5.0, 6.0, 5.0]

    # MSE minimization
    estimated_state_values = [0.0, 0.0]
    for _ in range(100000):
        original_step(belief_values, belief_distr, estimated_state_values)

    print(f"unbiased state estimates: {estimated_state_values}")
    for belief in belief_distr:
        print(
            f"unbiased value estimate of {belief} is {get_estimated_belief_value(belief, estimated_state_values)}"
        )

    # bias minimization
    estimated_state_values = [0.0, 0.0]
    for _ in range(100000):
        bias_step(belief_values, belief_distr, estimated_state_values)

    print(f"biased state estimates: {estimated_state_values}")
    for belief in belief_distr:
        print(
            f"biased value estimate of {belief} is {get_estimated_belief_value(belief, estimated_state_values)}"
        )
