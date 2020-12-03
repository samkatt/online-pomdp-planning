#!/usr/bin/env python

"""Tests for `online_pomdp_planning.mcts` module."""

import pytest  # type: ignore

from online_pomdp_planning.mcts import ActionNode, ObservationNode


def test_action_constructor():
    """Tests initiation of action nodes"""
    stats = (True, False, 10.0)
    p = ObservationNode()
    n = ActionNode(stats, p)

    assert stats == n.stats
    assert p == n.parent

    some_other_parent = ObservationNode()
    some_other_statistics = (1, 2, 3, 4)

    assert some_other_parent != n.parent
    assert some_other_statistics != n.stats


@pytest.mark.parametrize("observation", [((0)), (False), ((0, 1))])
def test_action_node_child(observation):
    """checks getting and setting child nodes"""
    root = ObservationNode()
    n = ActionNode(initial_statistics=None, parent=root)

    # if child not in node, do not allow fetching it
    with pytest.raises(KeyError):
        n.observation_node(observation)

    child = ObservationNode(parent=n)
    n.add_observation_node(observation, child)

    # cannot modify existing child
    with pytest.raises(AssertionError):
        n.add_observation_node(observation, child)

    # now child is in node, make sure the correct thing is returned
    assert child == n.observation_node(observation)


@pytest.mark.parametrize(
    "parent", [(None), (ActionNode("garbage statistic", ObservationNode()))]
)
def test_observation_node__constructor(parent):
    """Tests initiation of observation nodes"""
    n = ObservationNode(parent)
    assert parent == n.parent

    other_node = ActionNode("garbage statistic", ObservationNode())

    assert other_node != n.parent


@pytest.mark.parametrize("action", [((0)), (False), ((0, 1))])
def test_observation_node_child(action):
    """checks getting and setting child nodes"""
    n = ObservationNode()

    # if child not in node, do not allow fetching it
    with pytest.raises(KeyError):
        n.action_node(action)

    child = ActionNode("some statistic", parent=n)
    n.add_action_node(action, child)

    # cannot modify existing child
    with pytest.raises(AssertionError):
        n.add_action_node(action, child)

    # now child is in node, make sure the correct thing is returned
    assert child == n.action_node(action)


def test_observation_child_stats():
    """Tests getting children statistics"""
    node = ObservationNode()

    action_1 = -0.5
    child_1 = ActionNode((1, 2, 3), node)
    node.add_action_node(action_1, child_1)

    action_2 = True
    child_2 = ActionNode((True, False, ("garbage")), node)
    node.add_action_node(action_2, child_2)

    assert node.child_stats == {
        action_1: child_1.stats,
        action_2: child_2.stats,
    }
