"""Defines some types for ease of reading"""

import abc


class Action(abc.ABC):
    """The abstract type representing actions"""


class Observation(abc.ABC):
    """The abstract type representing observations"""


class State(abc.ABC):
    """The abstract type representing states"""


class Belief(abc.ABC):
    """The abstract type representing beliefs

    We expect the belief to sample states

    .. automethod:: __call__
    """

    @abc.abstractmethod
    def __call__(self) -> State:
        """Required implementation of belief: the ability to sample states"""


class Planner(abc.ABC):
    """The abstract class representation for planners in this package

    .. automethod:: __call__
    """

    @abc.abstractmethod
    def __call__(self, belief: Belief) -> Action:
        """
        The main functionality this package offers: a method that takes in a
        belief and returns an action

        :param belief:
        :type belief: Belief
        :return: the chosen action
        :rtype: Action
        """
