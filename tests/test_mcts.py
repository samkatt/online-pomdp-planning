#!/usr/bin/env python
"""tests for :mod:`online_pomdp_planning.mcts`"""

import random
from functools import partial
from math import log, sqrt
from operator import itemgetter
from typing import Counter, Dict

import numpy as np
import pytest

from online_pomdp_planning.mcts import (
    ActionNode,
    DeterministicNode,
    MuzeroInferenceOutput,
    ObservationNode,
    alphazero_scores,
    backprop_running_q,
    create_muzero_root,
    create_root_node_with_child_for_all_actions,
    deterministic_qval_backpropagation,
    expand_and_rollout,
    expand_node_with_all_actions,
    has_simulated_n_times,
    initiate_info,
    max_backup,
    max_q_action_selector,
    max_visits_action_selector,
    mc_backup,
    muzero_expand_node,
    muzero_scores,
    random_policy,
    rollout,
    select_action,
    select_deterministc_leaf_by_max_scores,
    select_leaf_by_max_scores,
    soft_q_action_selector,
    ucb_scores,
    unified_ucb_scores,
    visit_prob_action_selector,
)
from online_pomdp_planning.types import Action, Info, Observation
from online_pomdp_planning.utils import MovingStatistic


def test_initiate_info():
    """Tests :func:`~online_pomdp_planning.mcts.test_initiate_info`"""
    info = initiate_info()

    assert info["ucb_num_terminal_sims"] == 0
    assert info["mcts_num_action_nodes"] == 0
    assert info["iteration"] == 0
    # little cheat to easily check for equality, please don't do this in your own code
    assert str(info["ucb_tree_depth"]) == str(MovingStatistic())
    assert str(info["q_statistic"]) == str(MovingStatistic())


def test_action_constructor():
    """Tests initiation of action nodes"""
    stats = (True, False, 10.0)
    action = "my action"
    p = ObservationNode()
    n = ActionNode(action, stats, p)

    assert stats == n.stats
    assert p == n.parent
    assert action == n.action

    some_other_parent = ObservationNode()
    some_other_statistics = (1, 2, 3, 4)

    assert some_other_parent != n.parent
    assert some_other_statistics != n.stats


@pytest.mark.parametrize("observation", [((0)), (False), ((0, 1))])
def test_action_node_child(observation):
    """checks getting and setting child nodes"""
    root = ObservationNode()
    n = ActionNode(action="action", initial_statistics=None, parent=root)

    # if child not in node, do not allow fetching it
    with pytest.raises(KeyError):
        n.observation_node(observation)

    child = ObservationNode(observation=observation, parent=n)
    n.add_observation_node(child)

    # cannot add existing child
    with pytest.raises(AssertionError):
        n.add_observation_node(child)

    # now child is in node, make sure the correct thing is returned
    assert child == n.observation_node(observation)


@pytest.mark.parametrize(
    "obs,parent",
    [
        (None, None),
        ("o", ActionNode("garbage action", "garbage statistic", ObservationNode())),
    ],
)
def test_observation_node__constructor(obs, parent):
    """Tests initiation of observation nodes"""
    n = ObservationNode(obs, parent)

    assert parent == n.parent
    assert n.observation == obs

    other_node = ActionNode("action", "garbage statistic", ObservationNode())

    assert other_node != n.parent

    if obs is not None:
        with pytest.raises(AssertionError):
            ObservationNode(observation=obs)

    if parent is not None:
        with pytest.raises(AssertionError):
            ObservationNode(parent=parent)

    with pytest.raises(AssertionError):
        ObservationNode("some observation")


@pytest.mark.parametrize("action", [((0)), (False), ((0, 1))])
def test_observation_node_child(action):
    """checks getting and setting child nodes"""
    n = ObservationNode()

    # if child not in node, do not allow fetching it
    with pytest.raises(KeyError):
        n.action_node(action)

    child = ActionNode(action, "some statistic", parent=n)
    n.add_action_node(child)

    # cannot add existing child
    with pytest.raises(AssertionError):
        n.add_action_node(child)

    # now child is in node, make sure the correct thing is returned
    assert child == n.action_node(action)


def test_observation_child_stats():
    """Tests getting children statistics"""
    node = ObservationNode()

    action_1 = -0.5
    child_1 = ActionNode(action_1, (1, 2, 3), node)
    node.add_action_node(child_1)

    action_2 = True
    child_2 = ActionNode(action_2, (True, False, ("garbage")), node)
    node.add_action_node(child_2)

    assert node.child_stats == {
        action_1: child_1.stats,
        action_2: child_2.stats,
    }


def test_deterministic_node():
    """Tests :class:`DeterministicNode`"""

    root = DeterministicNode({"stat1": 1, "stat2": "bla"}, None)

    assert not root.expanded
    assert root.stats["stat1"] == 1
    assert root.child_stats == {}
    assert root.parent is None

    child = DeterministicNode({"childstat1": 2}, root)
    root.add_child("some_action", child)
    assert root.expanded
    assert not child.expanded

    assert root.child("some_action") == child
    assert root.parent is None
    assert child.parent == root

    with pytest.raises(KeyError):
        root.child("other action")

    assert root.stats["stat1"] == 1
    assert root.child_stats == {"some_action": child.stats}


def test_history():
    """Tests :meth:`online_pomdp_planning.mcts.ObservationNode.history`"""
    root = ObservationNode()

    first_action_node = ActionNode("first action", {"first"}, root)
    root.add_action_node(first_action_node)

    first_observation_node = ObservationNode("first observation", first_action_node)
    first_action_node.add_observation_node(first_observation_node)

    assert root.history() == []

    hist = first_observation_node.history()
    assert hist[0].action == "first action"
    assert hist[0].observation == "first observation"

    assert len(hist) == 1

    second_action_node = ActionNode(
        "second action", {"some stat: 0"}, first_observation_node
    )
    first_observation_node.add_action_node(second_action_node)

    second_observation_node = ObservationNode("second observation", second_action_node)
    second_action_node.add_observation_node(second_observation_node)

    hist = second_observation_node.history()

    assert hist[0].action == "first action"
    assert hist[0].observation == "first observation"
    assert hist[-1].action == "second action"
    assert hist[-1].observation == "second observation"

    assert len(hist) == 2


@pytest.mark.parametrize(
    "n,it,expectation", [(5, 4, False), (5, 5, True), (5, 6, True), (0, 0, True)]
)
def test_has_simulated_n_times(n, it, expectation):
    """Tests :func:`online_pomdp_planning.mcts.has_simulated_n_times`"""
    assert has_simulated_n_times(n, {"iteration": it}) == expectation


def test_has_simulated_n_times_asserts():
    """Tests :func:`online_pomdp_planning.mcts.has_simulated_n_times` assertions"""

    with pytest.raises(AssertionError):
        has_simulated_n_times(-1, {"iteration": 0})

    with pytest.raises(AssertionError):
        has_simulated_n_times(1, {"iteration": -1})

    with pytest.raises(KeyError):
        has_simulated_n_times(10, {"iteration_typo": 100})


@pytest.mark.parametrize(
    "actions,init_stats",
    [
        (
            [False, 1, (10, 2)],
            {
                False: {"some garbage": 100},
                1: {"more gargbage": False},
                (10, 2): {"HAH": "ha"},
            },
        ),
    ],
)
def test_create_root_node_with_child_for_all_actions(actions, init_stats):
    """Tests :func:`~online_pomdp_planning.mcts.create_root_node_with_child_for_all_actions`"""
    node = create_root_node_with_child_for_all_actions(None, {}, init_stats)

    for a in actions:
        assert node.action_node(a).stats == init_stats[a]
        assert node.action_node(a).parent == node
        assert node.action_node(a).observation_nodes == {}


@pytest.mark.parametrize(
    "actions,init_stats",
    [
        (
            [False, 1, (10, 2)],
            {
                False: {"some garbage": 100, "qval": 1.12},
                1: {"more gargbage": False, "qval": -12.234},
                (10, 2): {"HAH": "ha", "qval": 0.023},
            },
        ),
    ],
)
def test_create_root_node_with_child_for_all_actions_records_q(actions, init_stats):
    """Tests :func:`~online_pomdp_planning.mcts.create_root_node_with_child_for_all_actions`

    Make sure it adds q values to statistic
    """
    info = {"q_statistic": MovingStatistic()}
    create_root_node_with_child_for_all_actions(None, info, init_stats)

    assert info["q_statistic"].num == len(actions)


def test_create_muzero_root():
    """tests :func:`create_muzero_root`"""
    latent_state = "latent_state"
    reward = 1.2
    prior: Dict[Action, float] = {"a1": 0.2, "a3": 0.5, "a5": 0.3}
    noise_dirichlet_alpha = 10
    noise_exploration_fraction = 0.2

    root = create_muzero_root(
        latent_state,
        {},
        reward,
        prior,
        noise_dirichlet_alpha,
        noise_exploration_fraction,
    )

    assert root.stats["latent_state"] == latent_state
    assert root.stats["reward"] == reward
    assert root.stats["qval"] == 0
    assert root.stats["n"] == 0

    stats = root.child_stats

    assert len(stats) == 3

    assert pytest.approx(sum(x["prior"] for x in stats.values()), 1)
    for a, stat in stats.items():
        assert pytest.approx(stat["prior"]) != prior[a]

    for a, stat in stats.items():
        assert stat["qval"] == 0
        assert stat["n"] == 0
        assert stat["action"] == a

    # tests on prior and setting noise
    # little noise:
    root = create_muzero_root(
        latent_state, {}, reward, prior, noise_dirichlet_alpha, 0.000001
    )
    for a, stat in root.child_stats.items():
        assert pytest.approx(stat["prior"], rel=0.001) == prior[a]

    # much noise:
    root = create_muzero_root(latent_state, {}, reward, prior, 100000, 1)
    for a, stat in root.child_stats.items():
        assert pytest.approx(stat["prior"], rel=0.01) == 1 / 3


@pytest.mark.parametrize(
    "stats,max_a",
    [
        ({0: {"useless_stuff": None, "qval": 0.1}}, 0),
        ({0: {"qval": -0.1}}, 0),
        ({0: {"qval": 0.1, "some usless things": 100}, 10: {"qval": -0.1}}, 0),
        ({0: {"qval": 0.1}, 10: {"qval": 1}}, 10),
        ({True: {"qval": 100}, 0: {"qval": 0.1}, 10: {"qval": 1}}, True),
    ],
)
def test_max_q_action_selector(stats, max_a):
    """tests :func:~online_pomdp_planning.mcts.max_q_action_selector"""
    info = {}
    assert max_q_action_selector(stats, info) == max_a

    qvals = info["max_q_action_selector-values"]
    assert len(qvals) == len(stats)

    for a, s in stats.items():
        assert qvals[a] == s["qval"]


@pytest.mark.parametrize(
    "stats,max_a",
    [
        ({"max_a": {"qval": 1.0}}, "max_a"),
        ({"max_a": {"qval": -1.0}}, "max_a"),
        ({"max_a": {"qval": 20}, False: {"qval": -10.0}}, "max_a"),
        (
            {
                False: {"qval": 10.0},
                True: {"uselessstuff": 10, "qval": 10000},
                "a1": {"qval": -100},
            },
            True,
        ),
    ],
)
def test_soft_q_action_selector_deterministic(stats, max_a):
    """tests :func:~online_pomdp_planning.mcts.soft_q_action_selector for simple deterministic scenarios"""
    info = {}

    assert soft_q_action_selector(stats, info) == max_a

    assert len(info["soft_q_action_selector-probabilities"]) == len(stats)
    assert sum(
        list(info["soft_q_action_selector-probabilities"].values())
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "stats,sorted_actions",
    [
        ({"max_a": {"qval": 1.0}}, ["max_a"]),
        ({"max_a": {"qval": -1.0}, "min_a": {"qval": -2.0}}, ["max_a", "min_a"]),
        ({"max_a": {"qval": 0.45}, False: {"qval": -0.2}}, ["max_a", False]),
        (
            {
                False: {"qval": 1.34},
                True: {"uselessstuff": 8.12, "qval": 4.43},
                "a1": {"qval": -1.2},
            },
            [True, False, "a1"],
        ),
    ],
)
def test_soft_q_action_selector(stats, sorted_actions):
    """tests :func:~online_pomdp_planning.mcts.soft_q_action_selector for stochastic scenarios"""
    info = {}

    actions = Counter([soft_q_action_selector(stats, info) for _ in range(1000)])
    assert [v for v, _ in actions.most_common()] == sorted_actions


@pytest.mark.parametrize(
    "stats,max_a",
    [
        ({"max_a": {"n": -1}}, "max_a"),
        ({"max_a": {"n": 11}, False: {"n": 10}}, "max_a"),
        (
            {False: {"n": 10}, True: {"uselessstuff": 10, "n": 15}, "a1": {"n": 1}},
            True,
        ),
    ],
)
def test_max_visits_action_selector(stats, max_a):
    """tests :func:`max_visits_action_selector`"""

    info = {}
    assert max_visits_action_selector(stats, info) == max_a

    act_to_visits = info["visit_action_selector-counts"]

    assert len(act_to_visits) == len(stats)

    for a, s in stats.items():
        assert act_to_visits[a] == s["n"]


@pytest.mark.parametrize(
    "stats,tot,max_a",
    [
        ({"max_a": {"n": 1}}, 1, "max_a"),
        ({"max_a": {"n": 100}, False: {"n": 1}}, 101, "max_a"),
        (
            {False: {"n": 10}, True: {"uselessstuff": 10, "n": 10000}, "a1": {"n": 0}},
            10010,
            True,
        ),
    ],
)
def test_visit_prob_action_selector(stats, tot, max_a):
    """tests :func:`visit_prob_action_selector`"""

    info = {}
    assert visit_prob_action_selector(stats, info) == max_a

    act_to_visits = info["visit_action_selector-counts"]

    assert len(act_to_visits) == len(stats)
    assert act_to_visits[0][0] == max_a

    for a, n in act_to_visits:
        assert stats[a]["n"] == n

    acts_to_probs = info["visit_action_selector-probabilities"]
    assert acts_to_probs[0][0] == max_a

    for a, n in acts_to_probs:
        assert stats[a]["n"] / tot == n


@pytest.mark.parametrize(
    "o,actions,init_stats",
    [
        (
            10,
            [0, True, (10.0)],
            {0: {"q-value": 0, "n": 0}, True: {"garbage"}, (10.0): {"tupled garb"}},
        ),
        (
            10,
            [0, (10.0)],
            {0: {"qval": 10, "n": 0}, (10.0): {"some stuff": "garbage", "qval": 0.3}},
        ),
    ],
)
def test_expand_node_with_all_actions(o, actions, init_stats):
    """tests :func:~online_pomdp_planning.mcts.expand_node_with_all_actions"""
    parent = ObservationNode()
    stats = 0
    action = "action"
    node = ActionNode(action, stats, parent)

    info = {"mcts_num_action_nodes": 0, "q_statistic": MovingStatistic()}
    expand_node_with_all_actions(init_stats, o, node, info)

    expansion = node.observation_node(o)

    assert info["mcts_num_action_nodes"] == 1
    assert expansion.parent is node
    assert node.observation_node(o) is expansion
    assert node.observation_node(o).observation is o
    assert len(expansion.action_nodes) == len(actions)

    for a, n in expansion.action_nodes.items():
        assert len(n.observation_nodes) == 0
        assert n.parent == expansion
        assert n.action is a
        assert n.stats == init_stats[a]

    expected_number_of_qvals = len(actions) if "qval" in init_stats[actions[0]] else 0
    assert info["q_statistic"].num == expected_number_of_qvals

    # test that calling again will results in a no-operation
    expand_node_with_all_actions(init_stats, o, node, info)

    assert info["mcts_num_action_nodes"] == 1
    assert expansion.parent is node
    assert node.observation_node(o) is expansion
    assert len(expansion.action_nodes) == len(actions)

    for a, n in expansion.action_nodes.items():
        assert len(n.observation_nodes) == 0
        assert n.parent == expansion
        assert n.action is a
        assert n.stats == init_stats[a]

    assert info["q_statistic"].num == expected_number_of_qvals


def fake_muzero_recurrance_inference(
    state, action, value, reward, policy, latent_state
):
    """Just fakes doing inference in muzero"""
    return MuzeroInferenceOutput(value, reward, policy, latent_state)


def test_muzero_expand_node():
    """tests "py:func:`muzero_expand_node`"""
    info = {}
    root = DeterministicNode(
        {"latent_state": "root", "reward": 0.5, "n": 0, "qval": 0.0}, None
    )
    first_leaf = DeterministicNode(
        {"prior": 0.1, "action": "a1", "n": 3, "qval": 0.0}, root
    )
    root.add_child("a1", first_leaf)

    assert not first_leaf.expanded

    latent_state = "first_leaf_state"
    reward = -0.23
    value = 2.2
    policy = {"a1": 0.4, "a2": 0.6}
    returned_value = muzero_expand_node(
        first_leaf,
        info,
        partial(
            fake_muzero_recurrance_inference,
            value=value,
            reward=reward,
            policy=policy,
            latent_state=latent_state,
        ),
    )

    assert returned_value == value
    assert first_leaf.stats["latent_state"] == latent_state
    assert first_leaf.stats["reward"] == reward
    assert len(first_leaf.children) == 2

    for stats in first_leaf.child_stats.values():
        assert stats["n"] == 0
        assert stats["qval"] == 0

    for a in ["a1", "a2"]:
        assert first_leaf.child(a).stats["prior"] == policy[a]


def test_ucb_scores():
    """tests `func:ucb_scores`"""
    u = 50.3
    action_stats = {
        "a1": {"qval": 10, "n": 9},
        True: {"qval": 1, "n": 1},
        10: {"qval": 3, "n": 0},
    }
    action_scores = ucb_scores(action_stats, {}, u)

    assert {"a1", True, 10} == set(action_scores.keys())
    assert action_scores[10] == float("inf")
    assert action_scores[True] == 1 + 50.3 * sqrt(log(10) / 1)


def random_action_stats():
    """Returns a random set of action stats"""
    N = random.randint(1, 10)
    q = [random.random() * 100 for _ in range(N)]
    n = [random.randint(0, 10) for _ in range(N)]

    unnormalized_probabilities = [random.random() for _ in range(N)]
    total_prob = sum(unnormalized_probabilities)
    p = [prob / total_prob for prob in unnormalized_probabilities]

    u = random.random() * 10
    u2 = random.random() * 10

    return {i: {"qval": q[i], "n": n[i], "prior": p[i]} for i in range(N)}, u, u2


def check_ucb_dicts_equal(d1, d2):
    """Tests *approximately* whether `Any -> float` dictionaries are equal"""
    assert d1.keys() == d2.keys()
    for v1, v2 in zip(d1.values(), d2.values()):
        assert v1 == pytest.approx(v2)


def test_unified_ucb_scores():
    """Tests :func:`unified_ucb_scores`"""

    def normalize_q(q: float, q_stat: MovingStatistic) -> float:
        if q_stat.min < q_stat.max:
            return q_stat.normalize(q)

        return q

    def create_tradition_ucb_scores(u: float):
        return partial(
            unified_ucb_scores,
            get_q=lambda s, _: s["qval"],
            get_nominator=lambda N: np.log(N) if N > 0 else 0,
            get_expl_term=lambda nom, n: sqrt(nom / n) if n > 0 else float("inf"),
            get_prior=lambda _: 1,
            get_base_term=lambda _: u,
        )

    def create_alpha_zero_scores(u: float):
        return partial(
            unified_ucb_scores,
            get_q=lambda s, info: normalize_q(s["qval"], info["q_statistic"]),
            get_nominator=sqrt,
            get_expl_term=lambda nom, n: nom / (1 + n),
            get_prior=itemgetter("prior"),
            get_base_term=lambda _: u,
        )

    def create_muzero_scores(u: float, u2: float):
        return partial(
            unified_ucb_scores,
            get_q=lambda s, info: normalize_q(s["qval"], info["q_statistic"]),
            get_nominator=sqrt,
            get_expl_term=lambda nom, n: nom / (1 + n),
            get_prior=itemgetter("prior"),
            get_base_term=lambda N: u + np.log((1 + N + u2) / u2),
        )

    for _ in range(100):
        stats, u, u2 = random_action_stats()

        check_ucb_dicts_equal(
            create_tradition_ucb_scores(u)(stats, {}), ucb_scores(stats, {}, u)
        )

        q_stat = MovingStatistic()
        for s in stats.values():
            q_stat.add(s["qval"])
        info = {"q_statistic": q_stat}

        check_ucb_dicts_equal(
            create_alpha_zero_scores(u)(stats, info),
            alphazero_scores(stats, info, u),
        )

        check_ucb_dicts_equal(
            muzero_scores(stats, info, u, u2),
            create_muzero_scores(u, u2)(stats, info),
        )


def test_alphazero_ucb_scores():
    """tests :func:`alphazero_scores`"""
    stats = {
        "a1": {"qval": 0, "n": 0, "prior": 0.7},
        True: {"qval": 0, "n": 0, "prior": 0.1},
        False: {"qval": 0, "n": 2, "prior": 0.1},
        (0, "a4"): {"qval": 2.3, "n": 2, "prior": 0.1},
    }

    info = {"q_statistic": MovingStatistic()}

    scores = alphazero_scores(stats, info, 0.5)

    assert len(scores) == len(stats)
    assert pytest.approx(scores["a1"]) == 0.7 * 2 * 0.5
    assert pytest.approx(scores[True]) == 0.1 * 2 * 0.5
    assert pytest.approx(scores[False]) == 0.1 * (2 / 3) * 0.5
    assert pytest.approx(scores[(0, "a4")]) == 2.3 + 0.5 * (2 / 3) * 0.1

    info["q_statistic"].add(0)
    info["q_statistic"].add(2.3)

    scores = alphazero_scores(stats, info, 0.5)

    assert len(scores) == len(stats)
    assert pytest.approx(scores["a1"]) == 0.7 * 2 * 0.5
    assert pytest.approx(scores[True]) == 0.1 * 2 * 0.5
    assert pytest.approx(scores[False]) == 0.1 * (2 / 3) * 0.5
    assert pytest.approx(scores[(0, "a4")]) == 1.0 + 0.5 * (2 / 3) * 0.1


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
    """Tests :func:`~online_pomdp_planning.mcts.select_with_ucb`"""
    scoring_method = partial(ucb_scores, ucb_constant=u)
    assert select_action(stats, {}, scoring_method) == expected_action


def test_select_with_ucb_is_random():
    """Tests :func:`~online_pomdp_planning.mcts.select_with_ucb` is random"""
    # 2 == bla
    stats = {
        True: {"qval": 10, "n": 10000},
        2: {"qval": 9, "n": 1},
        "bla": {"qval": 9, "n": 1},
    }

    scoring_method = partial(ucb_scores, ucb_constant=10)
    chosen_actions = {select_action(stats, {}, scoring_method) for _ in range(20)}

    assert len(chosen_actions) == 2


def construct_ucb_tree(observation_from_simulator) -> ObservationNode:
    """Constructs a particular tree for UCB

    Tree: (action -> stats or obs)
        - ``False`` -> `(q=3.4, n=3)`:
            - ``True``
            - `(100)`
            - 2:
                - `(10, 2)` -> `(qval: 0, n: 0)`
        - 2 -> `(q=3.4, n=3)`

    According to UCB, the best first action is ``False``, the only second action is `(10, 2)`
    """
    root = ObservationNode()

    # two initial action nodes, action `False` is better
    better_first_action = False
    better_first_action_node = ActionNode(
        better_first_action, {"qval": 3.4, "n": 3}, root
    )
    worse_first_action = 2
    worse_first_action_node = ActionNode(
        worse_first_action, {"qval": -2.0, "n": 4}, root
    )

    root.add_action_node(better_first_action_node)
    root.add_action_node(worse_first_action_node)

    # three observation nodes; observation `2` is returned by simulator
    first_picked_observation_node = ObservationNode(
        observation_from_simulator, better_first_action_node
    )
    better_first_action_node.add_observation_node(first_picked_observation_node)
    better_first_action_node.add_observation_node(
        ObservationNode(True, better_first_action_node)
    )
    better_first_action_node.add_observation_node(
        ObservationNode((100), better_first_action_node)
    )

    # one leaf action node
    leaf_action_node = ActionNode(
        (10, 2), {"qval": 0, "n": 0}, first_picked_observation_node
    )
    better_first_action_node.observation_node(
        observation_from_simulator
    ).add_action_node(leaf_action_node)

    return root


def run_ucb_select_leaf(observation_from_simulator, root, max_depth=1000):
    """Runs UCB with a typical simulator from root"""

    def sim(s, a):
        """Fake simulator, returns state 0, obs 2, reward .5, not terminal, and info"""
        return 0, observation_from_simulator, 0.5, False

    info = {
        "leaf_depth": 0,
        "ucb_tree_depth": MovingStatistic(),
        "ucb_num_terminal_sims": 0,
    }
    scoring_method = partial(ucb_scores, ucb_constant=1)
    chosen_leaf, s, obs, term, rewards = select_leaf_by_max_scores(
        sim=sim,
        scoring_method=scoring_method,
        max_depth=max_depth,
        node=root,
        info=info,
        state=1,
    )
    return chosen_leaf, s, obs, term, rewards, info


def run_ucb_select_leaf_terminal_sim(observation_from_simulator, root):
    """Runs UCB with a terminal simulator from root"""

    def term_sim(s, a):
        """Returns the same as :func:`sim` but sets terminal flag to ``True``"""
        return 0, observation_from_simulator, 0.5, True

    info = {
        "leaf_depth": 0,
        "ucb_tree_depth": MovingStatistic(),
        "ucb_num_terminal_sims": 0,
    }
    scoring_method = partial(ucb_scores, ucb_constant=1)
    chosen_leaf, s, obs, term, rewards = select_leaf_by_max_scores(
        sim=term_sim,
        scoring_method=scoring_method,
        max_depth=1000,
        node=root,
        info=info,
        state=1,
    )
    return chosen_leaf, s, obs, term, rewards, info


def test_select_leaf_by_max_scores():
    """A specific test on UCB to see what leaf it returns"""

    observation_from_simulator = 2

    root = construct_ucb_tree(observation_from_simulator)

    chosen_leaf, s, obs, term, rewards, info = run_ucb_select_leaf(
        observation_from_simulator, root
    )

    leaf_action_node = root.action_node(False).observation_node(2).action_node((10, 2))

    assert chosen_leaf is leaf_action_node, "constructed tree should lead to leaf"
    assert s == 0, "simulator always outputs 0 as state"
    assert obs == observation_from_simulator, "better output the correct observation"
    assert not term, "simulator should indicate it is not terminal"
    assert rewards == [0.5, 0.5], "we did two steps of .5 reward"
    assert info["ucb_tree_depth"].max == 2
    assert info["ucb_num_terminal_sims"] == 0
    assert info["leaf_depth"] == 2

    # test max depth
    for d in [1, 2]:
        chosen_leaf, s, obs, term, rewards, info = run_ucb_select_leaf(
            observation_from_simulator, root, max_depth=d
        )
        assert info["ucb_tree_depth"].max == d
        assert info["leaf_depth"] == d
        assert info["ucb_num_terminal_sims"] == 0

    chosen_leaf, s, obs, term, rewards, info = run_ucb_select_leaf_terminal_sim(
        observation_from_simulator, root
    )

    assert chosen_leaf is root.action_node(
        False
    ), "constructed tree should lead to leaf"
    assert s == 0, "simulator always outputs 0 as state"
    assert obs == observation_from_simulator, "better output the correct observation"
    assert term, "simulator should indicate it is not terminal"
    assert rewards == [0.5], "we did two steps of .5 reward"
    assert info["leaf_depth"] == 1


def test_select_deterministc_leaf_by_max_scores():
    """Some tests on :func:`select_deterministc_leaf_by_max_scores`"""
    node_scoring_method = partial(ucb_scores, ucb_constant=10)
    info = {"ucb_tree_depth": MovingStatistic()}

    # if only one leaf, should find it
    root = DeterministicNode(
        {"latent_state": "root", "reward": 0.5, "n": 0, "qval": 0.0}, None
    )
    first_leaf = DeterministicNode(
        {"prior": 0.1, "action": "a1", "n": 3, "qval": 0.0}, root
    )
    root.add_child("a1", first_leaf)
    assert select_deterministc_leaf_by_max_scores(node_scoring_method, root, info) == (
        first_leaf,
        None,
    )
    assert info["ucb_tree_depth"].max == 1

    # a second, better, leaf should be picked instead
    second_leaf = DeterministicNode(
        {"prior": 0.1, "action": "a2", "n": 3, "qval": 5.0}, root
    )
    root.add_child("a2", second_leaf)
    assert select_deterministc_leaf_by_max_scores(node_scoring_method, root, info) == (
        second_leaf,
        None,
    )
    assert info["ucb_tree_depth"].max == 1
    assert info["ucb_tree_depth"].num == 2

    # trying to add more nodes, should pick it
    third_leaf = DeterministicNode(
        {"prior": 0.1, "action": "a", "n": 3, "qval": -5.0}, second_leaf
    )
    second_leaf.add_child("s", third_leaf)
    assert select_deterministc_leaf_by_max_scores(node_scoring_method, root, info) == (
        third_leaf,
        None,
    )
    assert info["ucb_tree_depth"].max == 2

    # increasing q value of first (bad) leaf should make it favourable
    first_leaf.stats["qval"] = 10000
    assert select_deterministc_leaf_by_max_scores(node_scoring_method, root, info) == (
        first_leaf,
        None,
    )
    assert info["ucb_tree_depth"].max == 2
    assert info["ucb_tree_depth"].num == 4


def test_max_backup():
    """Tests that :func:`~online_pomdp_planning.mcts.max_backup` raises bad discount"""
    root = ObservationNode()

    actions = [True, "some action", 0, ("stuff", 2)]
    qs = [(random.random() - 0.5) * 10 for _ in range(len(actions))]

    # generate some statistics
    for a, q in zip(actions, qs):
        root.add_action_node(ActionNode(a, {"qval": q}, root))

    assert max_backup(
        random.choice(list(root.action_nodes.values())), random.random()
    ) == max(qs)


def test_backprop_running_q_assertion():
    """Tests that :func:`~online_pomdp_planning.mcts.backprop_running_q` raises bad discount"""
    root = ObservationNode()
    leaf = ActionNode("some action", {"qval": 0.3, "n": 1}, root)

    root.add_action_node(leaf)

    for backup_operator in [mc_backup, max_backup]:

        with pytest.raises(AssertionError):
            backprop_running_q(
                leaf,
                [1.0],
                0,
                {"q_statistic": MovingStatistic()},
                -1,
                backup_operator,
            )
        with pytest.raises(AssertionError):
            backprop_running_q(
                leaf, [1.0], 0, {"q_statistic": MovingStatistic()}, 1.1, backup_operator
            )

        # ``None`` is acceptable leaf evaluation
        backprop_running_q(
            leaf,
            [1.0],
            None,
            {"q_statistic": MovingStatistic()},
            0.9,
            backup_operator,
        )


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
    """Tests :func:`~online_pomdp_planning.mcts.backprop_running_q`"""
    observation_from_simulator = 2
    root = construct_ucb_tree(observation_from_simulator)

    # fake leaf node
    leaf_node = root.action_node(False).observation_node(2).action_node((10, 2))

    leaf_selection_output = [0.1, 7.0]
    leaf_evaluation = -5
    backprop_running_q(
        leaf_node,
        leaf_selection_output,
        leaf_evaluation,
        {"q_statistic": MovingStatistic()},
        discount_factor,
        mc_backup,
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


def test_deterministic_qval_backpropagation():
    """Tests :func:`deterministic_qval_backpropagation"""
    q_statistic = MovingStatistic()
    q_statistic.add(5)
    q_statistic.add(-1)
    info = {"q_statistic": q_statistic}

    # create tree
    root = DeterministicNode(
        {"latent_state": "root", "reward": 0.5, "n": 0, "qval": 0.0}, None
    )
    first_leaf = DeterministicNode(
        {"prior": 0.1, "action": "a1", "n": 3, "qval": 0.0, "reward": 0}, root
    )
    root.add_child(first_leaf.stats["action"], first_leaf)
    second_leaf = DeterministicNode(
        {"prior": 0.9, "action": "a2", "n": 4, "qval": 5.0, "reward": 0.25}, first_leaf
    )
    first_leaf.add_child(second_leaf.stats["action"], second_leaf)

    deterministic_qval_backpropagation(0.9, second_leaf, None, 9.75, info)

    assert info["q_statistic"].max > 5
    assert info["q_statistic"].min == -1

    assert (
        root.stats["n"] == 1
        and first_leaf.stats["n"] == 4
        and second_leaf.stats["n"] == 5
    )

    # (5 * 4 + 9.75 + .25) / 5
    assert second_leaf.stats["qval"] == 6.0
    # return = (9.75 + 0.25) * .9 = 9, (3 * 0 + 9) / 4 = 2.25
    assert first_leaf.stats["qval"] == 2.25
    # return = 9 * .9 + 0.5 = ..., ... / 1
    assert root.stats["qval"] == 9 * 0.9 + 0.5

    # Quick test to see that ``None`` is acceptable as leaf eval
    deterministic_qval_backpropagation(0.9, second_leaf, None, None, info)


def test_rollout():
    """Tests :func:`~online_pomdp_planning.mcts.rollout`"""

    def pol(s, o) -> Action:
        return random_policy([False, 1, (10, 2)], s, o)

    discount_factor = 0.9
    depth = 3
    state = 1
    obs = 0

    def sim(s, a):
        """Fake simulator, returns state 0, obs 2, reward .5 and not terminal"""
        return 0, 2, 0.5, False

    def term_sim(s, a):
        """Returns the same as :func:`sim` but sets terminal flag to ``True``"""
        return 0, 2, 0.5, True

    assert rollout(pol, term_sim, 0, discount_factor, state, obs) == 0

    assert (
        rollout(pol, term_sim, depth, discount_factor, state, obs) == 0.5
    ), "terminal sim should allow 1 action"

    assert (
        rollout(pol, sim, 2, discount_factor, state, obs) == 0.5 + discount_factor * 0.5
    ), "1 depth should allow 1 action"


def test_expand_and_rollout():
    """Tests :func:`expansion`"""

    def pol(s, o) -> Action:
        return random_policy([False, 1, (10, 2)], s, o)

    discount_factor = 0.9
    depth = 3
    state = 1
    obs = 0

    def sim(s, a):
        """Fake simulator, returns state 0, obs 2, reward .5 and not terminal"""
        return 0, 2, 0.5, True

    expansion_results = {}

    def save_expansion(o: Observation, a: ActionNode, info: Info):
        """Mock expansion"""
        expansion_results["obs"] = o
        expansion_results["leaf"] = a
        expansion_results["info"] = info

    ret = expand_and_rollout(
        save_expansion, pol, sim, depth, discount_factor, "leaf!", state, obs, {}
    )

    assert expansion_results["leaf"] == "leaf!"
    assert expansion_results["obs"] == obs
    assert expansion_results["info"] == {}

    assert ret == 0.5


if __name__ == "__main__":
    pytest.main([__file__])
