"""Defines some types for ease of reading"""

from typing import Any, Dict, Hashable, Tuple

from typing_extensions import Protocol

Action = Hashable
"""The abstract type representing actions requires to be hash-able"""
Observation = Hashable
"""The abstract type representing observations requires to be hash-able"""
State = Any
"""The abstract type for a state, no particular protocol is expected"""


class Simulator(Protocol):
    """The abstract type representing simulators

    We expect simulators to map a state and action into a next state,
    observation, reward and terminal signal

    .. automethod:: __call__
    """

    def __call__(self, s: State, a: Action) -> Tuple[State, Observation, float, bool]:
        """Simulate a transition

        state, action -> next state, observation, reward, terminal generator

        Args:
            s (State): an input state
            a (Action): a chosen action

        Returns:
            Tuple[State, Observation, float, bool]:
        """


class Belief(Protocol):
    """The abstract type representing beliefs

    We expect the belief to sample states

    .. automethod:: __call__
    """

    def __call__(self) -> State:
        """Required implementation of belief: the ability to sample states"""


Info = Dict[str, Any]
"""Data type used for information flow from implementation to caller"""


class Planner(Protocol):
    """The abstract class representation for planners in this package

    .. automethod:: __call__
    """

    def __call__(self, belief: Belief) -> Tuple[Action, Info]:
        """
        The main functionality this package offers: a method that takes in a
        belief and returns an action

        :param belief:
        :return: the chosen action and run-time information
        """
