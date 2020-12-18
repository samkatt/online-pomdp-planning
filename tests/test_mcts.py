#!/usr/bin/env python
"""tests for `online_pomdp_planning.mcts` module."""

from functools import partial
from math import log, sqrt

import pytest  # type: ignore

from online_pomdp_planning.mcts import (
    ActionNode,
    ObservationNode,
    backprop_running_q,
    create_root_node_with_child_for_all_actions,
    expand_node_with_all_actions,
    pick_max_q,
    random_policy,
    rollout,
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
    "actions,init_stats",
    [
        ([False, 1, (10, 2)], "some garbage"),
        ([], {"qval": 10, "n": 0}),
    ],
)
def test_create_root_node_with_child_for_all_actions(actions, init_stats):
    """Tests :py:func:`~online_pomdp_planning.mcts.create_root_node_with_child_for_all_actions`"""
    node = create_root_node_with_child_for_all_actions(actions, init_stats)

    for a in actions:
        assert node.action_node(a).stats == init_stats
        assert node.action_node(a).parent == node
        assert node.action_node(a).observation_nodes == {}


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

    expand_node_with_all_actions(actions, init_stats, o, node, {})

    expansion = node.observation_node(o)

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
                - (10, 2) -> (qval: 0, n: 0)
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
    leaf_action_node = ActionNode({"qval": 0, "n": 0}, first_picked_observation_node)
    better_first_action_node.observation_node(
        observation_from_simulator
    ).add_action_node((10, 2), leaf_action_node)

    return root


def run_ucb_select_leaf(observation_from_simulator, root):
    """Runs UCB with a typical simulator from root"""

    def sim(_, __):
        """Fake simulator, returns state 0, obs 2, reward .5 and not terminal"""
        return 0, observation_from_simulator, 0.5, False

    chosen_leaf, s, obs, term, rewards = ucb_select_leaf(
        sim=sim, ucb_constant=1, state=1, node=root, info={}
    )
    return chosen_leaf, s, obs, term, rewards


def run_ucb_select_leaf_terminal_sim(observation_from_simulator, root):
    """Runs UCB with a terminal simulator from root"""

    def term_sim(_, __):
        """Returns the same as :py:func:`sim` but sets terminal flag to `True`"""
        return 0, observation_from_simulator, 0.5, True

    chosen_leaf, s, obs, term, rewards = ucb_select_leaf(
        sim=term_sim, ucb_constant=1, state=1, node=root, info={}
    )
    return chosen_leaf, s, obs, term, rewards


def test_ucb_select_leaf():
    """A specific test on UCB to see what leaf it returns"""

    observation_from_simulator = 2

    root = construct_ucb_tree(observation_from_simulator)

    chosen_leaf, s, obs, term, rewards = run_ucb_select_leaf(
        observation_from_simulator, root
    )

    leaf_action_node = root.action_node(False).observation_node(2).action_node((10, 2))

    assert chosen_leaf is leaf_action_node, "constructed tree should lead to leaf"
    assert s == 0, "simulator always outputs 0 as state"
    assert obs == observation_from_simulator, "better output the correct observation"
    assert not term, "simulator should indicate it is not terminal"
    assert rewards == [0.5, 0.5], "we did two steps of .5 reward"

    chosen_leaf, s, obs, term, rewards = run_ucb_select_leaf_terminal_sim(
        observation_from_simulator, root
    )

    assert chosen_leaf is root.action_node(
        False
    ), "constructed tree should lead to leaf"
    assert s == 0, "simulator always outputs 0 as state"
    assert obs == observation_from_simulator, "better output the correct observation"
    assert term, "simulator should indicate it is not terminal"
    assert rewards == [0.5], "we did two steps of .5 reward"


def test_backprop_running_q_assertion():
    """Tests that :py:func:`~online_pomdp_planning.mcts.backprop_running_q` raises bad discount"""
    some_obs_node = ObservationNode()
    with pytest.raises(AssertionError):
        backprop_running_q(-1, ActionNode("gargabe", some_obs_node), [], 0, {})
    with pytest.raises(AssertionError):
        backprop_running_q(1.1, ActionNode("gargabe", some_obs_node), [], 0, {})


@pytest.mark.parametrize(
    "discount_factor, new_q_first, new_q_leaf",
    [
        (0, 10.3 / 4, 7.0),
        (1, 12.3 / 4, 2),
        # hard math, let's not do that again (3.4*3 + .1 + .9* 7 + .9*.9*-5)
        (0.9, 12.55 / 4, 7 - 4.5),
    ],
)
def test_backprop_running_q(discount_factor, new_q_first, new_q_leaf):
    """Tests :py:func:`~online_pomdp_planning.mcts.backprop_running_q`"""
    observation_from_simulator = 2
    root = construct_ucb_tree(observation_from_simulator)

    # fake leaf node
    leaf_node = root.action_node(False).observation_node(2).action_node((10, 2))

    leaf_selection_output = [0.1, 7.0]
    leaf_evaluation = -5
    backprop_running_q(
        discount_factor, leaf_node, leaf_selection_output, leaf_evaluation, {}
    )

    # lots of math by hand, hope this never needs to be re-computed
    # basically we _know_ the path taken, the rewards, and the original tree
    # so we can compute what the updated q-values and 'n' are
    # q-values are running average, 'n' is just incremented

    assert leaf_node.stats["n"] == 1
    assert leaf_node.stats["qval"] == pytest.approx(new_q_leaf)

    first_chosen_action_node = root.action_node(False)

    assert first_chosen_action_node.stats["qval"] == pytest.approx(new_q_first)
    assert first_chosen_action_node.stats["n"] == 4


def test_rollout():
    """Tests :py:func:`~online_pomdp_planning.mcts.rollout`"""

    pol = partial(random_policy, ([False, 1, (10, 2)]))
    discount_factor = 0.9
    depth = 3
    terminal = False
    state = 1
    obs = 0

    def sim(_, __):
        """Fake simulator, returns state 0, obs 2, reward .5 and not terminal"""
        return 0, 2, 0.5, False

    def term_sim(_, __):
        """Returns the same as :py:func:`sim` but sets terminal flag to `True`"""
        return 0, 2, 0.5, True

    assert (
        rollout(pol, term_sim, depth, discount_factor, state, obs, t=True, info={}) == 0
    )
    assert rollout(pol, term_sim, 0, discount_factor, state, obs, terminal, {}) == 0

    assert (
        rollout(pol, term_sim, depth, discount_factor, state, obs, terminal, {}) == 0.5
    ), "terminal sim should allow 1 action"

    assert (
        rollout(pol, sim, 2, discount_factor, state, obs, terminal, {})
        == 0.5 + discount_factor * 0.5
    ), "1 depth should allow 1 action"
