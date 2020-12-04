#!/usr/bin/env python
"""tests for `online_pomdp_planning.mcts` module."""

from math import log, sqrt

import pytest  # type: ignore

from online_pomdp_planning.mcts import (
    ActionNode,
    ObservationNode,
    expand_node_with_all_actions,
    pick_max_q,
    select_with_ucb,
    ucb,
    ucb_select_leaf,
)


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


@pytest.mark.parametrize(
    "stats,max_a",
    [
        ({0: {"qval": 0.1}}, 0),
        ({0: {"qval": -0.1}}, 0),
        ({0: {"qval": 0.1}, 10: {"qval": -0.1}}, 0),
        ({0: {"qval": 0.1}, 10: {"qval": 1}}, 10),
        ({True: {"qval": 100}, 0: {"qval": 0.1}, 10: {"qval": 1}}, True),
    ],
)
def test_maxq(stats, max_a):
    """tests :py:func:~online_pomdp_planning.mcts.pick_max_q"""
    assert pick_max_q(stats) == max_a


@pytest.mark.parametrize(
    "o,actions,init_stats",
    [
        (10, [0, True, (10.0)], {"q-value": 0, "n": 0}),
        (10, [0, (10.0)], {"q-value": 10, "n": 0}),
    ],
)
def test_expand_node_with_all_actions(o, actions, init_stats):
    """tests :py:func:~online_pomdp_planning.mcts.expand_node_with_all_actions"""
    parent = ObservationNode()
    stats = 0
    node = ActionNode(stats, parent)

    expansion = expand_node_with_all_actions(o, node, actions, init_stats)

    assert expansion.parent is node
    assert node.observation_node(o) is expansion
    assert len(expansion.action_nodes) == len(actions)

    for n in expansion.action_nodes.values():
        assert len(n.observation_nodes) == 0
        assert n.parent == expansion
        assert n.stats == init_stats
        assert n.stats is not init_stats  # please be copy


@pytest.mark.parametrize(
    "q,n,n_total,ucb_constant,expected_raise",
    [
        (123, 0, 234, 452, False),
        (0, 0, -234, False, True),
        (0, -1, 10, False, True),
        (0, 1, 1, 0, False),
        (-5.2, 1, 1, 1, False),
    ],
)
def test_ucb_raises(q, n, n_total, ucb_constant, expected_raise):
    """Tests that :py:func:`~online_pomdp_planning.mcts.ucb` raises on invalid input"""
    if expected_raise:
        with pytest.raises(AssertionError):
            ucb(q, n, n_total, ucb_constant)
    else:
        ucb(q, n, n_total, ucb_constant)


@pytest.mark.parametrize(
    "q,n,n_total,ucb_constant,expectation",
    [
        (123, 0, 234, 452, float("inf")),
        (0, 1, 1, 1, sqrt(log(1) / 1)),
        (-5.2, 1, 1, 1, -5.2 + sqrt(log(1) / 1)),
        (134, 3, 4, 1, 134 + sqrt(log(4) / 3)),
        (1, 1, 1, 50.3, 1 + 50.3 * sqrt(log(1) / 1)),
        (1, 1, 10, 50.3, 1 + 50.3 * sqrt(log(10) / 1)),
    ],
)
def test_ucb(q, n, n_total, ucb_constant, expectation):
    """Tests :py:func:`~online_pomdp_planning.mcts.ucb`"""
    assert ucb(q, n, n_total, ucb_constant) == expectation


@pytest.mark.parametrize(
    "expected_action,u,stats",
    [
        (True, 0, {True: {"qval": 10, "n": 10000}, 2: {"qval": 9, "n": 1}}),
        (2, 1, {True: {"qval": 10, "n": 10000}, 2: {"qval": 9, "n": 1}}),
        (
            (1, 2),
            1,
            {
                True: {"qval": 10, "n": 10000},
                2: {"qval": 9, "n": 1},
                (1, 2): {"qval": 10, "n": 1},
            },
        ),
    ],
)
def test_select_with_ucb(expected_action, u, stats):
    """Tests :py:func:`~online_pomdp_planning.mcts.select_with_ucb`"""
    assert select_with_ucb(stats, u) == expected_action


def construct_ucb_tree(observation_from_simulator) -> ObservationNode:
    """Constructs a particular tree for UCB

    Tree: (action -> stats or obs)
        - False -> (q=3.4, n=3):
            - True
            - (100)
            - 2:
                - (10, 2) -> (qval: -10, n: 100)
        - 2 -> (q=3.4, n=3)

    According to UCB, the best first action is `False`, the only second action is `(10, 2)`
    """
    root = ObservationNode()

    # two initial action nodes, action `False` is better
    better_first_action = False
    better_first_action_node = ActionNode({"qval": 3.4, "n": 3}, root)
    worse_first_action = 2
    worse_first_action_node = ActionNode({"qval": -2.0, "n": 4}, root)

    root.add_action_node(better_first_action, better_first_action_node)
    root.add_action_node(worse_first_action, worse_first_action_node)

    # three observation nodes; observation `2` is returned by simulator
    first_picked_observation_node = ObservationNode(better_first_action_node)
    better_first_action_node.add_observation_node(
        observation_from_simulator, first_picked_observation_node
    )
    better_first_action_node.add_observation_node(
        True, ObservationNode(better_first_action_node)
    )
    better_first_action_node.add_observation_node(
        (100), ObservationNode(better_first_action_node)
    )

    # one leaf action node
    leaf_action_node = ActionNode(
        {"qval": -10, "n": 100}, first_picked_observation_node
    )
    better_first_action_node.observation_node(
        observation_from_simulator
    ).add_action_node((10, 2), leaf_action_node)

    return root


def test_ucb_select_leaf():
    """A specific test on UCB to see what leaf it returns"""

    observation_from_simulator = 2

    root = construct_ucb_tree(observation_from_simulator)
    leaf_action_node = root.action_node(False).observation_node(2).action_node((10, 2))

    def sim(_, __):
        """Fake simulator, returns state 0, obs 2, reward .5 and not terminal"""
        return 0, observation_from_simulator, 0.5, False

    chosen_leaf, s, obs, term, rewards = ucb_select_leaf(
        state=1, node=root, sim=sim, ucb_constant=1
    )

    assert chosen_leaf is leaf_action_node, "constructed tree should lead to leaf"
    assert s == 0, "simulator always outputs 0 as state"
    assert obs == observation_from_simulator, "better output the correct observation"
    assert not term, "simulator should indicate it is not terminal"
    assert rewards == [0.5, 0.5], "we did two steps of .5 reward"

    def term_sim(_, __):
        """Returns the same as :py:func:`sim` but sets terminal flag to `True`"""
        s, o, r, _ = sim(None, None)
        return s, o, r, True

    chosen_leaf, s, obs, term, rewards = ucb_select_leaf(
        state=1, node=root, sim=term_sim, ucb_constant=1
    )

    assert chosen_leaf is root.action_node(
        False
    ), "constructed tree should lead to leaf"
    assert s == 0, "simulator always outputs 0 as state"
    assert obs == observation_from_simulator, "better output the correct observation"
    assert term, "simulator should indicate it is not terminal"
    assert rewards == [0.5], "we did two steps of .5 reward"
