"""Defines some types for ease of reading"""

from typing import Protocol


class Action(Protocol):
    """The abstract type representing actions

    Requires to be hash-able

    """

    def __hash__(self):
        """Can be hashed"""


class Observation(Protocol):
    """The abstract type representing observations

    Requires to be hash-able

    """

    def __hash__(self):
        """Can be hashed"""


class State(Protocol):
    """The abstract type representing states"""


class Belief(Protocol):
    """The abstract type representing beliefs

    We expect the belief to sample states

    .. automethod:: __call__
    """

    def __call__(self) -> State:
        """Required implementation of belief: the ability to sample states"""


class Planner(Protocol):
    """The abstract class representation for planners in this package

    .. automethod:: __call__
    """

    def __call__(self, belief: Belief) -> Action:
        """
        The main functionality this package offers: a method that takes in a
        belief and returns an action

        :param belief:
        :type belief: Belief
        :return: the chosen action
        :rtype: Action
        """