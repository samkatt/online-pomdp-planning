"""Implementation of Monte-Carlo tree search"""
from __future__ import annotations

import random
from functools import partial
from math import isclose, log, sqrt
from operator import xor
from timeit import default_timer as timer
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from typing_extensions import Protocol

from online_pomdp_planning.types import (
    Action,
    ActionObservation,
    Belief,
    History,
    Info,
    Observation,
    Planner,
    Simulator,
    State,
)
from online_pomdp_planning.utils import MovingStatistic

Stats = Mapping[str, Any]
"""Alias type for statistics: a mapping from some description to anything"""
ActionStats = Mapping[Action, Stats]
"""Alias type for action statistics: a mapping from actions to :class:`Stats`"""


def initiate_info() -> Info:
    """Simple initiation of info object, populating expected values

    Returns a dictionary with starter values for:
        - ucb_num_terminal_sims
        - mcts_num_action_nodes
        - ucb_tree_depth
        - q_statistic
        - iteration
    """
    return {
        "ucb_num_terminal_sims": 0,
        "mcts_num_action_nodes": 0,
        "ucb_tree_depth": MovingStatistic(),
        "q_statistic": MovingStatistic(),
        "iteration": 0,
    }


class ActionNode:
    """A decision node in the MCTS tree

    This node maps observations to children nodes. In addition it stores
    statistics, such as the expected Q-value associated with this node.
    """

    def __init__(
        self,
        action: Action,
        initial_statistics: Any,
        parent: ObservationNode,
    ):
        """Initializes the action node with given statistics

        :param action: the action associated with this node
        :param initial_statistics: anything you would like to store
        :param parent: the parent node in the tree
        """
        self.action = action
        self.stats = initial_statistics
        self.parent = parent
        self.observation_nodes: Dict[Observation, ObservationNode] = {}

    def add_observation_node(self, observation_node: ObservationNode):
        """Adds a node to the children of `self`

        Raises ``AssertionError`` if:
            - the parent of the added node is not self
            - ``observation`` is not already associated with a child node

        :param observation_node: the new child node
        """
        assert observation_node.observation not in self.observation_nodes
        assert observation_node.parent == self
        self.observation_nodes[observation_node.observation] = observation_node

    def observation_node(self, observation: Observation) -> ObservationNode:
        """The child-node associated with given observation ``o``

        Raises ``KeyError`` if ``action`` is not associated with a child node

        :param observation:
        :return: child node
        """
        return self.observation_nodes[observation]

    def __repr__(self) -> str:
        """Pretty print action nodes"""
        return f"ActionNode({self.action}, {self.stats}, {self.parent})"


class ObservationNode:
    """A chance/observation node in the MCTS tree

    This node representation action-observation history, ending up in a
    particular observation. It has no statistics. If it has no parent, then it
    is the root node. This node maps actions to children nodes.
    """

    def __init__(
        self,
        observation: Optional[Observation] = None,
        parent: Optional[ActionNode] = None,
    ):
        """Initiates an observation node with given parent

        Either *no argument* should be given, or _both_, will otherwise throw
        and assertion error.

        If None are given, it must be the root of a tree.

        :param observation: observation associated with this node
        :param parent: if no parent is given, this must be the root node
        """
        # ugly because SE is hard, but check
        # to make sure either none or both arguments are given
        assert not xor(observation is None, parent is None)

        self.observation = observation
        self.parent = parent
        self.action_nodes: Dict[Action, ActionNode] = {}

    @property
    def child_stats(self) -> ActionStats:
        """Returns a mapping from actions to statistics (shortcut)

        :return: action -> stats mapping
        """
        return {a: n.stats for a, n in self.action_nodes.items()}

    def add_action_node(self, node: ActionNode):
        """Adds a ``action`` -> ``node`` mapping to children

        Raises ``AssertionError`` if:
            - the parent of the added node is not `self`
            - ``action`` is not already associated with a child node

        :param node: child node
        """
        assert node.action not in self.action_nodes
        assert node.parent == self
        self.action_nodes[node.action] = node

    def action_node(self, action: Action) -> ActionNode:
        """Get child node associated with ``action``

        Raises ``KeyError`` if ``action`` is not associated with a child node

        :param action:
        :return: returns child node
        """
        return self.action_nodes[action]

    def history(self) -> History:
        """Returns the history from root to `self`

        Implemented recursively: returns empty list if root, otherwise parent's
        history + self

        :return: list of action-observation tuples (first element is for t = 0)
        """
        if self.observation is None:
            # self is root
            return []

        assert self.parent is not None

        # recurrent call on parent,
        # and append then current observation and parent's action
        return self.parent.parent.history() + [
            ActionObservation(self.parent.action, self.observation)
        ]

    def __repr__(self) -> str:
        """Pretty print observation nodes"""
        return f"ObservationNode({self.observation}, {self.parent})"


class DeterministicNode:
    """A node in the tree of a deterministic process

    While most MCTS methods assume stochastic dynamics, if the environment is
    deterministic, then there is one branching less to care about. This node
    allows for constructing such trees.

    Specifically, the children of `self` are themselves
    :class:`DeterministicNode`. Each action has one 'outcome' and thus one
    node associated with it.
    """

    def __init__(self, stats: Any, parent: Optional[DeterministicNode]):
        """Initiates the node and sets its ``parent`` and ``stats``

        :param stats: stored, can be anything that is used by algorithms traversing the tree
        :param parent: the parent of this node, if null then must be root
        """
        self.stats = stats
        self.parent = parent
        self.children: Dict[Action, DeterministicNode] = {}

    @property
    def expanded(self) -> bool:
        """Whether `self` has been expanded yet"""
        return len(self.children) > 0

    @property
    def child_stats(self) -> ActionStats:
        """Returns an action => stats dictionary of the children"""
        return {a: n.stats for a, n in self.children.items()}

    def add_child(self, a: Action, n: DeterministicNode):
        """Adds ``n`` as child associated with ``a``"""
        assert a not in self.children
        assert n.parent == self
        self.children[a] = n

    def child(self, a: Action) -> DeterministicNode:
        """Get child node associated with ``a``"""
        return self.children[a]


class StopCondition(Protocol):
    """The protocol for a stop condition during MCTS

    Determines, given :class:`online_pomdp_planning.types.Info`, whether to
    continue simulating or not.

    .. automethod:: __call__
    """

    def __call__(self, info: Info) -> bool:
        """Signature for the stop condition

        Determines, given info, whether to stop or not

        :param info: run time information
        :return: ``True`` if determined stop condition is met
        """
        raise NotImplementedError()


def no_stop(info: Info) -> bool:
    """:class:`StopCondition` implementatio that always returns ``False``"""
    return False


def has_simulated_n_times(n: int, info: Info) -> bool:
    """Returns true if number of iterations in ``info`` exceeds ``n``

    Given ``n``, implements :class:`StopCondition`

    :param n: number to have iterated
    :param info: run time info (assumed to have entry "iteration" -> int)
    :return: true if number of iterations exceed ``n``
    """
    assert n >= 0 and info["iteration"] >= 0

    return n <= info["iteration"]


class ProgressBar(StopCondition):
    """A :class:`StopCondition` call that prints out a progress bar

    Note: Always returns ``False``, and meant to be used in combination with
    other stop condition

    The progress bar is printed by `tqdm`, and will magnificently fail if
    something else is printed or logged during.

    XXX: not tested because UI is hard to test, please modify with care

    .. automethod:: __call__
    """

    def __init__(self, max_sims: int):
        """Sets up a progress bar for up to ``max_sims`` simulations

        We assume that ``max_sims`` will be _exactly_ the number
        of samples to be accepted (i.e. calls to ``__call__``). Any less will
        not close the progress bar, any more and the progress bar will reset.

        :param max_sims: 'length' of progress bar
        """
        assert max_sims >= 0

        super().__init__()
        self._max_sims = max_sims

        # `tqdm` starts the progress bar upon initiation. At this point the
        # belief update is not happening yet, so we do not want to print it
        self.pbar: Optional[tqdm] = None

    def __call__(self, info: Info) -> bool:
        """Updates the progression bar

        Initiates the bar when `info["iteration"]` is 0, closes when
        ``self._max_sims`` is reached

        :param info: run time information (assumed to map "iteration" -> int)
        :return: _always_ ``False``
        """
        current_sim = info["iteration"]
        assert current_sim >= 0

        if current_sim == 0:
            # the first sample is accepted, LGTM
            self.pbar = tqdm(total=self._max_sims)

        assert self.pbar
        self.pbar.update()

        if current_sim >= self._max_sims - 1:
            # last sample accepted!
            self.pbar.close()

        return False


class LeafSelection(Protocol):
    """The signature for leaf selection

    .. automethod:: __call__
    """

    def __call__(
        self, s: State, node: ObservationNode, info: Info
    ) -> Tuple[ActionNode, State, Observation, bool, Any]:
        """Traverses through the tree and picks a leaf

        :param node: (root) node
        :param info: run time information
        :return: leaf node, state and obs, terminal flag and input to :class:`BackPropagation`
        """
        raise NotImplementedError()


class DeterministicLeafSelection(Protocol):
    """The signature for leaf selection

    .. automethod:: __call__
    """

    def __call__(
        self, node: DeterministicNode, info: Info
    ) -> Tuple[DeterministicNode, Any]:
        """Traverses through the tree and picks a leaf

        :param node: (root) node
        :param info: run time information
        :return: leaf node and input to :class:`DeterministicBackPropagation`
        """
        raise NotImplementedError()


ActionScoringMethod = Callable[[ActionStats, Info], Dict[Action, float]]
"""Type used to evaluate actions during tree traversal"""


def ucb_scores(
    stats: ActionStats,
    info: Info,
    ucb_constant: float,
) -> Dict[Action, float]:
    """The upper-confidence bound scoring method (used in :func:`select_action`)

    Assumes that ``stats`` contains an entry for "qval" and "n".

    Given ``ucb_constant``, implements the :class:`ActionScoringMethod`.

    UCB is `q + ucb_constant * sqrt(log(log_n_total) / n)`.

    :param stats: an action => stats mapping
    :param info: ignored
    :param ucb_constant: the upper-confidence bound constant
    :return: an action => score mapping, here the upper confidence bound on the q values
    """

    total_visits = sum(s["n"] for s in stats.values())

    # special case where no action has ever been taken
    if total_visits == 0:
        return {a: float("inf") for a in stats}

    log_total_visits = log(total_visits)

    def ucb(q, n):
        if n == 0:
            return float("inf")

        return q + ucb_constant * sqrt(log_total_visits / n)

    return {a: ucb(s["qval"], s["n"]) for a, s in stats.items()}


def alphazero_scores(
    stats: ActionStats,
    info: Info,
    ucb_constant: float,
) -> Dict[Action, float]:
    """The UCB scoring method combined with a prior

    Returns an action => score mapping, where the scores are::

        norm(q) + ucb_constant * prior * (sqrt(N) / 1 + n)
        q       + prior & exploration

    with `N` being total number of visits and `n` being the number of visits of
    the child node, and `norm(q)` is the normalized q value of the node.

    Assumes ``stats`` contains "qval", "prior", and "n" for every action and
    that ``info`` contains "q_statistic".

    Given ``ucb_constant``, implements the :class:`ActionScoringMethod`.

    Taken from AlphaZero::

        Silver, David, et al. "Mastering the game of go without human
        knowledge." nature 550.7676 (2017): 354-359.

    :param stats: action => stats mapping
    :param info: contains "q_staticstic"
    :param ucb_constant: first exploration constant
    :return: action => float scores
    """
    # unpacking some stuff
    q_stat = info["q_statistic"]
    total_visits_sqrt = sqrt(sum(s["n"] for s in stats.values()))

    if q_stat.min < q_stat.max:
        q_values = {a: q_stat.normalize(s["qval"]) for a, s in stats.items()}
    else:
        q_values = {a: s["qval"] for a, s in stats.items()}

    def pucb(q, n, p):
        return q + ucb_constant * p * (total_visits_sqrt / (1 + n))

    return {a: pucb(q_values[a], s["n"], s["prior"]) for a, s in stats.items()}


def muzero_scores(
    stats: ActionStats, info: Info, c1: float, c2: float
) -> Dict[Action, float]:
    """The UCB scoring method used my muzero

    Returns an action => score mapping, where the scores are::

        norm(q) + prior * (sqrt(N) / 1 + n) * (c1 + log((N + c2 + 1) / c2))
        q       + prior & exploration       *  base term

    with `N` being total number of visits and `n` being the number of visits of
    the child node, and `norm(q)` is the normalized q value of the node.

    Assumes ``stats`` contains "qval", "n", and "prior" for every action and
    that ``info`` contains "q_statistic".

    Given ``c1`` and ``c2``, implements the :class:`ActionScoringMethod`.

    *Not* tested, no idea how to sensibly do that honestly.

    See paper Schrittwieser, Julian, et al. Mastering atari, go, chess and
    shogi by planning with a learned model." Nature 588.7839 (2020): 604-609.".

    :param stats: action => stats mapping
    :param info: contains "q_staticstic"
    :param c1: first exploration constant
    :param c2: second exploration constant
    :return: action => float scores
    """
    assert c1 > 0 and c2 > 0
    q_stat = info["q_statistic"]

    # pre-computed for all actions
    total_visits = sum(s["n"] for s in stats.values())

    # base term
    base_term = c1 + np.log((total_visits + c2 + 1) / c2)

    # q: assigning `0` to unvisited actions (not "inf")
    q_values = {a: stat["qval"] for a, stat in stats.items()}
    if q_stat.min < q_stat.max:
        q_values = {a: q_stat.normalize(q) for a, q in q_values.items()}

    total_visits_sqrt = sqrt(total_visits)  # pre-computation

    priors = {
        a: stat["prior"] * (total_visits_sqrt / (1 + stat["n"]))
        for a, stat in stats.items()
    }

    # q     +    prior & expl   *   base term
    return {a: q_values[a] + priors[a] * base_term for a in stats}


def unified_ucb_scores(
    stats: ActionStats,
    info: Info,
    get_q: Callable[[Stats, Info], float],
    get_nominator: Callable[[float], float],
    get_expl_term: Callable[[float, float], float],
    get_prior: Callable[[Stats], float],
    get_base_term: Callable[[float], float],
):
    """A unified UCB scoring method

    I basically got sick of all the different ways that UCB can be computed,
    and wanted to unify these in one function, see:

        - :func:`ucb_scores`
        - :func:`alphazero_scores`
        - :func:`muzero_scores`

    So here we are, the resulting scores will be:

        get_q(stat, info) + get_prior(stat) * get_expl_term(get_nominator(N), stat) * get_base_term(N)

    For example:

        >>> def normalize_q(q: float, q_stat: MovingStatistic) -> float:
                if q_stat.min < q_stat.max:
                    return q_stat.normalize(q)
                return q

        >>> ucb_scores = return partial(
                unified_ucb_scores,
                get_q=lambda s, _: s["qval"],
                get_nominator=lambda N: np.log(N) if N > 0 else 0,
                get_expl_term=lambda nom, n: sqrt(nom / n) if n > 0 else float('inf'),
                get_prior=lambda _: 1,
                get_base_term=lambda _: u,
            )

        >>> alphazero_scores = return partial(
                unified_ucb_scores,
                get_q=lambda s, info: normalize_q(s["qval"], info["q_statistic"]),
                get_nominator=sqrt,
                get_expl_term=lambda nom, n: nom / (1 + n),
                get_prior=itemgetter("prior"),
                get_base_term=lambda _: u,
            )

        >>> muzero_scores = return partial(
                unified_ucb_scores,
                get_q=lambda s, info: normalize_q(s["qval"], info["q_statistic"]),
                get_nominator=sqrt,
                get_expl_term=lambda nom, n: nom / (1 + n),
                get_prior=itemgetter("prior"),
                get_base_term=lambda N: u + np.log((1 + N + u2) / u2),
            )

    :param stats: action => stats mapping, contains keys dependent on other input
    :param info: run time information, can contain "q_statistic" if ``get_q`` needs it
    :param get_q: a function that returns the q value given 'stats' and ``info``
    :param get_nominator: how to compute the nominator given to ``get_expl_term``
    :param get_expl_term: how to compute the exploration term given nominator and 'n'
    :param get_prior: how to get the prior (if at all) given stats
    :param get_base_term: how to compute the 'base_term' (see mu-zero)
    """
    # gather statistics they all need
    N = sum(s["n"] for s in stats.values())

    nom = get_nominator(N)
    b = get_base_term(N)

    return {
        a: get_q(stat, info) + get_prior(stat) * get_expl_term(nom, stat["n"]) * b
        for a, stat in stats.items()
    }


def select_action(
    stats: ActionStats,
    info: Info,
    scoring_method: ActionScoringMethod,
) -> Action:
    """Select an action using ``scoring_method``

    Exactly how each action is scored (given ``stats``) is up to the scoring
    method. This function simply picks a randomly between the actions that are
    given the maximum score.

    Given ``scoring_method``, implements :class:`ActionSelection`

    :param stats: the statistics: Action -> Dict where Dict is {"qval": float, "n": int}
    :param info: current MCTS running info, given to ``scoring_method``
    :param scoring_method: the method that transforms ``stats`` into scores
    :return: the action with the highest upper confidence bound
    """
    action_scores = scoring_method(stats, info)

    max_score = max(action_scores.values())
    return random.choice(
        [a for a, score in action_scores.items() if score >= max_score]
    )


def select_leaf_by_max_scores(
    sim: Simulator,
    scoring_method: ActionScoringMethod,
    max_depth: int,
    state: State,
    node: ObservationNode,
    info: Info,
) -> Tuple[ActionNode, State, Observation, bool, List[float]]:
    """Tree policy: ``scoring_method`` to pick actions, simulator to generate observations

    When provided with ``sim`` and ``scoring_method``, it implements
    :class:`LeafSelection`, and returns a list of rewards as output

    Picks action nodes according to :func:`select_action` --- with
    ``scoring_method`` --- uses ``sim`` to generate observations and pick the
    respective nodes

    Tracks the tree depth, maintains a running statistic on it in ``info``
    (``ucb_tree_depth``), and stops going down the tree when ``max_depth`` is
    reached. Additionally sets ``leaf_depth``

    NOTE::

        Assumes "ucb_num_terminal_sims", "ucb_tree_depth" and "leaf_depth" to
        be entries in ``info``

    See :func:`unified_ucb_scores` for ways of constructing ``scoring_method``

    :param sim: a POMDP simulator
    :param scoring_method: function that, given action stats, returns their scores
    :param max_depth: max length of the tree to go down
    :param state: the root state
    :param node: the root node
    :param info: run time information (ignored)
    :return: leaf node, state, observation, terminal flag and list of rewards
    """
    assert max_depth > 0
    list_of_rewards: List[float] = []

    depth = 0
    while True:

        action = select_action(node.child_stats, info, scoring_method)
        state, obs, reward, terminal_flag = sim(state, action)

        list_of_rewards.append(reward)
        depth += 1

        if terminal_flag or depth >= max_depth:
            break

        try:
            node = node.action_node(action).observation_node(obs)
        except KeyError:
            # action node is a leaf
            break

    # info tracking number of terminal simulations
    if terminal_flag:
        info["ucb_num_terminal_sims"] += 1

    # info tracking tree depth
    info["ucb_tree_depth"].add(depth)
    info["leaf_depth"] = depth

    return node.action_node(action), state, obs, terminal_flag, list_of_rewards


def select_deterministc_leaf_by_max_scores(
    scoring_method: ActionScoringMethod,
    node: DeterministicNode,
    info: Info,
) -> Tuple[DeterministicNode, Any]:
    """Deterministic tree policy according to ``scoring_method``

    When provided with ``scoring_method``, it implements
    :class:`DeterministicLeafSelection`.

    Returns "None" as second argument

    Picks action nodes according to ``scoring_method``.

    Tracks the tree depth, maintains a running statistic on it in
    `info["ucb_tree_depth"]`.

    NOTE::

        Assumes "ucb_tree_depth" is an entry in ``info``

    :param scoring_method: function that, given action stats, returns their scores
    :param node: the root node
    :param info: run time information
    :return: leaf node
    """

    depth = 0

    while node.expanded:
        action = select_action(node.child_stats, info, scoring_method)
        node = node.child(action)
        depth += 1

    # info tracking tree depth
    info["ucb_tree_depth"].add(depth)

    return node, None


class Expansion(Protocol):
    """The signature for leaf node expansion


    A 'helper' interface for the :class:`ExpandAndEvaluate` interface, making
    it easy for various expansion and evaluation methods to be combined since
    they are often separate concers.

    .. automethod:: __call__
    """

    def __call__(self, o: Observation, action_node: ActionNode, info: Info):
        """Expands action_node leaf node

        :param o: observation that resulted in leaf
        :param action_node: action that resulted in leaf
        :param info: run time information
        :return: nothing, modifies the tree
        """


class DeterministicNodeExpansion(Protocol):
    """The signature for leaf node expansion

    .. automethod:: __call__
    """

    def __call__(self, n: DeterministicNode, info: Info) -> Any:
        """Expands a leaf node

        :param n: current (not-expanded) leaf node
        :param info: run time information
        :return: evaluation, can be whatever, given to :class:`DeterministicBackPropagation`
        """


def expand_node_with_all_actions(
    action_stats: ActionStats,
    o: Observation,
    action_node: ActionNode,
    info: Info,
):
    """Adds an observation node to the tree with action_node child for each action

    Expands ``action_node`` with new :class:`ObservationNode` with action
    child for each :class:`~online_pomdp_planning.types.Action`

    When provided with the  initial stats, this implements :class:`Expansion`

    NOTE: ``action_node`` must not have action_node child node associated with
    ``o`` or this will result in action_node no-operation.

    NOTE: requires ``info`` to contain entry for "mcts_num_action_nodes"

    :param action_stats: the initial statistics for each node
    :param o: the new observation
    :param action_node: the current leaf node
    :param info: run time information -- requires "mcts_num_action_nodes"
    :return: modifies tree
    """
    if o in action_node.observation_nodes:
        return

    if len(action_node.observation_nodes) == 0:
        # first time this action node was expanded,
        # so now we count it as part of the tree
        info["mcts_num_action_nodes"] += 1

    expansion = ObservationNode(observation=o, parent=action_node)

    for a, stats in action_stats.items():
        expansion.add_action_node(ActionNode(a, stats, expansion))

    # cheap test, assuming if one statistic contains "qval", then all do (and visa-versa)
    if "q_statistic" in info and "qval" in next(iter(action_stats.values())):
        for stats in action_stats.values():
            info["q_statistic"].add(stats["qval"])

    action_node.add_observation_node(expansion)


def muzero_expand_node(
    n: DeterministicNode,
    info: Info,
    inference: Callable[[State, Action], MuzeroInferenceOutput],
) -> float:
    """Muzero's way of expanding and evaluating a node

    ``n`` is assumed to contain "action", and its parent "latent_state".

    Will create a child for each action in the policy evaluated through
    ``inference`` and create a child for it.

    :param n: node to be expanded
    :param info: ignored
    :param inference: function used to get latent state and evaluations
    """
    assert n.parent
    assert not n.expanded

    network_output = inference(n.parent.stats["latent_state"], n.stats["action"])

    n.stats["latent_state"] = network_output.latent_state
    n.stats["reward"] = network_output.reward

    n.children = {
        a: DeterministicNode({"prior": p, "action": a, "n": 0, "qval": 0.0}, n)
        for a, p in network_output.policy.items()
    }

    return network_output.value


class ExpandAndEvaluate(Protocol):
    """The signature of leaf node evaluation

    Expansion and evaluation are often separate concerns, so many of the
    methods implementing this interface may directly accept an
    :class:`Expansion` to easily combine different expansion and evaluation
    techniques.

    .. automethod:: __call__
    """

    def __call__(self, leaf: ActionNode, s: State, o: Observation, info: Info) -> Any:
        """Evaluates a leaf node

        :param leaf: leaf to expand and evaluate
        :param s: state to evaluate
        :param o: observation to evaluate
        :param info: run time information
        :return: evaluation, can be whatever, given to :class:`BackPropagation`
        """


class Policy(Protocol):
    """The signature for a policy"""

    def __call__(self, s: State, o: Observation) -> Action:
        """A (stochastic) mapping from state and/or observation to action

        :param s: the current state
        :param o: the current observation
        :return: an action
        """


def random_policy(actions: Sequence[Action], _: State, __: Observation) -> Action:
    """A random policy just picks a random action

    Implements :class:`Policy` given ``actions``

    :param actions: list of actions to pick randomly from
    :param _: ignored (state)
    :param __: ignored (observation)
    :return: a random action
    """
    return random.choice(actions)


def expand_and_rollout(
    expansion_strategy: Expansion,
    policy: Policy,
    sim: Simulator,
    depth: int,
    discount_factor: float,
    leaf: ActionNode,
    s: State,
    o: Observation,
    info: Info,
) -> float:
    """Expands ``leaf`` according to ``expansion_strategy`` and evaluate with ``policy``

    Given ``expansion_strategy``, ``policy``, ``sim``, ``depth``, and
    ``discount_factor``, this implements :class:`ExpandAndEvaluate` where it
    returns a float (discounted return) as metric.

    Calls :func:`rollout` after calling ``expansion_strategy```

    When the terminal flag ``t`` is set, this function will return 0 and does
    not expand.

    ``expand_and_rollout`` does not really care about how ``leaf`` is expanded, it
    accepts any :class:`Expansion`. Will expand ``leaf`` before performing expand_and_rollout.

    :param expansion_strategy: how to expand ``leaf``
    :param policy: expand_and_rollout policy
    :param sim: a POMDP simulator
    :param depth: the longest number of actions to take
    :param discount_factor: discount factor of the problem
    :param leaf: leaf to expand and evaluate
    :param s: starting state
    :param o: starting observation
    :param t: whether the episode has terminated
    :param info: run time information (ignored)
    :return: the discounted return of following ``policy`` in ``sim``
    """
    assert 0 <= discount_factor <= 1
    assert depth >= 0, "prevent never ending loop"

    if depth == 0:
        return 0.0

    expansion_strategy(o, leaf, info)
    return rollout(policy, sim, depth, discount_factor, s, o)


def rollout(
    policy: Policy,
    sim: Simulator,
    depth: int,
    discount_factor: float,
    s: State,
    o: Observation,
) -> float:
    """Do a rollout in ``sim`` starting from ``s`` following ``policy``

    Runs ``policy`` in ``sims`` until some depth or terminal transition and
    returns the (discounted) return.

    :param policy: expand_and_rollout policy
    :param sim: a POMDP simulator
    :param depth: the longest number of actions to take
    :param discount_factor: discount factor of the problem
    :param s: starting state
    :param o: starting observation
    :return: the discounted return of following ``policy`` in ``sim``
    """
    assert 0 <= discount_factor <= 1
    assert depth >= 0, "prevent never ending loop"

    ret = 0.0

    discount = 1.0
    for _ in range(depth):
        a = policy(s, o)
        s, o, r, t = sim(s, a)

        ret += r * discount
        discount *= discount_factor

        if t:
            break

    return ret


def expand_and_evaluate_with_model(
    leaf: ActionNode,
    s: State,
    o: Observation,
    info: Info,
    model: Callable[
        [Optional[ActionNode], State, Optional[Observation], Info],
        Tuple[float, ActionStats],
    ],
) -> float:
    """Evaluates ``leaf`` through ``model`` on ``s`` and stores prior

    Assumes ``model`` returns both the (discounted return) value of ``s`` and a
    prior policy (mapping from action to its probability)

    Will expand a new :class:`ObservationNode` under ``leaf`` with a child for
    each action (prior) outputted by the ``moddel``, returns the predicted
    value.

    Given ``model`` implements :class:`ExpandAndEvaluate`.

    Inspired from AlphaZero::

        Silver, David, et al. "Mastering the game of go without human
        knowledge." nature 550.7676 (2017): 354-359.

    :param leaf: leaf to expand
    :param s: state to be evaluated by ``model``
    :param o: observation to be evaluated (potentially) by ``model``
    :param t: if ``true``, will cause this function to return zeros
    :param info: provided to ``model``, otherwise ignored
    :param model: used to evaluate leaf to generate stats and value to return
    :return: the value as predicted by ``model``
    """
    v, stats = model(leaf, s, o, info)
    expand_node_with_all_actions(stats, o, leaf, info)

    return v


class BackPropagation(Protocol):
    """The signature for back propagation through nodes

    .. automethod:: __call__
    """

    def __call__(
        self,
        n: ActionNode,
        leaf_selection_output: Any,
        leaf_eval_output: Optional[Any],
        info: Info,
    ) -> None:
        """Updates the nodes visited during selection

        :param n: The leaf node that was expanded
        :param leaf_selection_output: The output of the selection method
        :param leaf_eval_output: The output of the evaluation method, if at all
        :param info: run time information
        :return: has only side effects
        """


class DeterministicBackPropagation(Protocol):
    """The signature for back propagation through deterministic nodes

    .. automethod:: __call__
    """

    def __call__(
        self,
        n: DeterministicNode,
        leaf_selection_output: Any,
        leaf_eval_output: Any,
        info: Info,
    ) -> None:
        """Updates the nodes visited during selection

        :param n: The leaf node that was expanded
        :param leaf_selection_output: The output of the selection method
        :param leaf_eval_output: The output of the evaluation method
        :param info: run time information
        :return: has only side effects
        """


def mc_backup(_: ActionNode, q: float) -> float:
    """Monte-Carlo backup operator: returns q"""
    return q


def max_backup(n: ActionNode, _: float) -> float:
    """Returns max Q-value of parent node"""
    return max(s["qval"] for s in n.parent.child_stats.values())


def backprop_running_q(
    leaf: ActionNode,
    leaf_selection_output: List[float],
    leaf_evaluation: Optional[float],
    info: Info,
    discount_factor: float,
    backup_operator: Callable[[ActionNode, float], float],
) -> None:
    """Updates running Q average of visited nodes

    Implements :class:`BackPropagation` given ``discount_factor`` and
    ``backup_operator``

    Updates the visited nodes (through parents of ``leaf``) by updating the
    running Q average. Assumes the statistics in nodes have mappings "qval" ->
    float and "leaf" -> int.

    Given a ``discount_factor``, implements :class:`BackPropagation` with a
    list of rewards as input from :class:`LeafSelection` and a return estimate
    (float) from :class:`ExpandAndEvaluate`.

    Will close bounds `info["q_statistic"]`

    Possible ``backup_operator`` include:

        - :func:`mc_backup`: just backs up sample q
        - :func:`max_backup`: backs up max q-value

    :param discount_factor: 'gamma' of the POMDP environment [0, 1]
    :param leaf: leaf node
    :param leaf_selection_output: list of rewards from tree policy
    :param leaf_evaluation: return estimate, assumed 0 if ``None``
    :param info: run time information (ignored)
    :param backup_operator: computes target_q for next backup
    :return: has only side effects
    """
    assert 0 <= discount_factor <= 1

    target_return = leaf_evaluation if leaf_evaluation else 0

    # loop through all rewards in reverse order
    # simultaneously traverse back up the tree through `leaf`
    n: Optional[ActionNode] = leaf
    for reward in reversed(leaf_selection_output):
        assert n, "somehow got to root without processing all rewards"

        target_return = reward + discount_factor * target_return

        # grab current stats
        stats = n.stats
        q, num = stats["qval"], stats["n"]

        # store next stats
        stats["qval"] = (q * num + target_return) / (num + 1)
        stats["n"] = num + 1

        # adjust bounds in `info`
        info["q_statistic"].add(stats["qval"])

        # figure out which value to back up
        target_return = backup_operator(n, target_return)

        # go up in tree
        n = n.parent.parent

    assert n is None, "somehow processed all rewards yet not reached root"


def deterministic_qval_backpropagation(
    discount_factor: float,
    leaf: DeterministicNode,
    leaf_selection_output: Any,
    leaf_eval_output: Optional[float],
    info: Info,
) -> None:
    """Backpropagation for deterministic trees (used in muzero)

    Implements :class:`DeterministicBackPropagation` given

    Will close bounds `info["q_statistic"]`

    :param discount_factor: discount factor used in computing return
    :param n: the leaf node to start propagating back from
    :param leaf_selection_output: ignored
    :param leaf_eval_output: expected to be a float (evaluation), assumed 0 if ``None``
    :param info: "q_statistic" updated
    :return: none, only side efects in tree
    """
    assert 0 <= discount_factor <= 1

    value = leaf_eval_output if leaf_eval_output else 0

    n: Optional[DeterministicNode] = leaf

    while n:

        value += n.stats["reward"]

        # update stats in `n`
        num, q = n.stats["n"], n.stats["qval"]

        n.stats["qval"] = (q * num + value) / (num + 1)
        n.stats["n"] = num + 1

        # adjust bounds in `info`
        info["q_statistic"].add(n.stats["qval"])

        value *= discount_factor
        n = n.parent


class ActionSelection(Protocol):
    """The signature for selection actions given the root node

    .. automethod:: __call__
    """

    def __call__(self, stats: ActionStats, info: Info) -> Action:
        """Selects the preferred action given statistics


        :param stats: statistics of the (root) node
        :param info: run time information
        :return: preferred action
        """


def max_q_action_selector(stats: ActionStats, info: Info) -> Action:
    """Picks the action with the highest 'q-value' in their statistics

    Assumes stats has a "qval" attribute

    Implements :class:`ActionSelection`

    Adds ranking to `info["max_q_action_selector-values"]`, which is a sorted
    list (by q-value) of action-stats pairs

    :param stats: assumes a "q" property in the statistic
    :param info: run time information (adds "max_q_action_selector-values")
    :return: action with highest q value
    """
    qvals = {k: v["qval"] for k, v in stats.items()}
    info["max_q_action_selector-values"] = qvals

    return select_action(stats, info, lambda _, __: qvals)


def soft_q_action_selector(stats: ActionStats, info: Info) -> Action:
    """Samples action through softmax on their q-values

    Assumes stats has a "qval" attribute

    Implements :class:`ActionSelection`

    Adds softmax probabilities to
    `info["soft_q_action_selector-probabilities"]`


    :param stats: assumes a "q" property in the statistic
    :param info: run time information (adds "soft_q_action_selector-probabilities")
    :return: sample action according to ~ softmax(q)
    """
    soft_q = softmax([s["qval"] for s in stats.values()])
    info["soft_q_action_selector-probabilities"] = dict(zip(stats, soft_q))

    return random.choices(list(stats.keys()), soft_q)[0]


def max_visits_action_selector(stats: ActionStats, info: Info) -> Action:
    """implements :class:`ActionSelection`. Samples action most picked by MCTS.

    Assumes ``stats`` is a action => dict statistic dictionary. Each of those
    dictionaries is expected to contain a "n" entry that reflects how often
    the action has been chosen.

    Populates `info["visit_action_selector-counts"]` with visit counts

    """
    action_visits = {k: v["n"] for k, v in stats.items()}
    info["visit_action_selector-counts"] = action_visits

    return select_action(stats, info, lambda _, __: action_visits)


def visit_prob_action_selector(stats: ActionStats, info: Info) -> Action:
    """implements :class:`ActionSelection`. Samples action according to visitation counts

    Assumes ``stats`` is a action => dict statistic dictionary. Each of those
    dictionaries is expected to contain a "n" entry that reflects how often
    the action has been chosen.

    Populates `info["visit_action_selector-probabilities"]` with probability
    distribution and `info["visit_action_selector-counts"]` with the actual
    visits.
    """
    # extract (and sort by) visits statistic
    action_visits = [(k, v["n"]) for k, v in stats.items()]
    info["visit_action_selector-counts"] = sorted(
        action_visits,
        key=lambda pair: pair[1],
        reverse=True,
    )

    # compute (sorted) probabilities
    tot = sum(visits for (_, visits) in info["visit_action_selector-counts"])
    assert tot > 0

    info["visit_action_selector-probabilities"] = [
        (a, visits / tot) for (a, visits) in info["visit_action_selector-counts"]
    ]

    return random.choices(
        [x[0] for x in info["visit_action_selector-probabilities"]],
        [x[1] for x in info["visit_action_selector-probabilities"]],
    )[0]


class TreeConstructor(Protocol):
    """The signature for creating the root node

    .. automethod:: __call__
    """

    def __call__(self, belief: Belief, info: Info) -> ObservationNode:
        """Creates a root node out of nothing

        :return: The root node
        """
        raise NotImplementedError()


class DeterministicTreeConstructor(Protocol):
    """The signature for creating the root node

    .. automethod:: __call__
    """

    def __call__(self, history_representation: Any, info: Info) -> DeterministicNode:
        """Creates a root node out of nothing

        :return: The root node
        """
        raise NotImplementedError()


def create_root_node_with_child_for_all_actions(
    belief: Belief,
    info: Info,
    action_stats: ActionStats,
) -> ObservationNode:
    """Creates a tree by initiating the first action nodes

    Implements :class:`TreeConstructor` given ``action_stats``

    :param belief: ignored
    :param info: ignored
    :param action_stats: the initial statistics of those nodes
    :return: the root of the tree
    """
    root = ObservationNode()

    for a, stats in action_stats.items():
        root.add_action_node(ActionNode(a, stats, root))

    # cheap test, assuming if one statistic contains "qval", then all do (and visa-versa)
    if "q_statistic" in info and "qval" in next(iter(action_stats.values())):
        for stats in action_stats.values():
            info["q_statistic"].add(stats["qval"])

    return root


def create_muzero_root(
    latent_state: Any,
    info: Info,
    reward: float,
    prior: Mapping[Action, float],
    noise_dirichlet_alpha: float,
    noise_exploration_fraction: float,
) -> DeterministicNode:
    """Creates a root node for mu-zero

    Given all input, implements :class:`DeterministicTreeConstructor`

    The prior value given to each child of the returned root is a weighted
    combination of ``prior`` and some noise. The variance of the noise is given
    by ``noise_dirichlet_alpha``, which is the parameter of a Dirichlet. The
    _larger_ this value, the higher the noise. The ``noise_exploration_fraction``
    is the weight given to the noise.

    :param latent_state: the current history/observation/state representation for muzero dynamics
    :param info: ignored
    :param reward: reward associated with current history/observation/state
    :param prior: action -> probability mapping of current history/observation/state
    :param noise_dirichlet_alpha: level of noise added to ``prior`` of root children
    :param noise_exploration_fraction: trade off between ``prior`` and noise
    """

    assert 0 <= noise_exploration_fraction <= 1
    assert isclose(1, sum(p for p in prior.values())), f"Prior is weird: {prior}"

    root = DeterministicNode(
        {"latent_state": latent_state, "reward": reward, "n": 0, "qval": 0.0}, None
    )

    # (noisy) prior in root for extra exploration
    noise_gen = np.random.default_rng().dirichlet([noise_dirichlet_alpha] * len(prior))

    # magic stolen from mu-zero:
    # each prior is given some noise.
    # Amount of variance in noise is set through `noise_dirichlet_alpha`
    # basically weighted average (according to `noise_exploration_fraction`
    noisy_prior = {
        a: p * (1 - noise_exploration_fraction)
        + noise_gen[i] * noise_exploration_fraction
        for i, (a, p) in enumerate(prior.items())
    }

    root.children = {
        a: DeterministicNode({"prior": p, "action": a, "n": 0, "qval": 0.0}, root)
        for a, p in noisy_prior.items()
    }

    return root


def mcts(
    stop_cond: StopCondition,
    tree_constructor: TreeConstructor,
    leaf_select: LeafSelection,
    expand_and_eval: ExpandAndEvaluate,
    backprop: BackPropagation,
    action_select: ActionSelection,
    horizon: int,
    belief: Belief,
) -> Tuple[Action, Info]:
    """The general MCTS method, defined by its components

    MCTS will simulate until ``stop_cond`` returns ``False``, where each
    simulation:

    #. Selects a leaf (action) node through ``leaf_select``
    #. Expands and evaluates the leaf node through ``expand_and_eval``
    #. Back propagates and updates node values through ``backprop``

    After spending the simulation budget, it picks an given the statistics
    stored in the root node through ``action_select``.

    The root node constructor allows for custom ways of initiating the tree

    During run time will maintain information
    :class`online-pomdp-planning.types.Info`, with "iteration" -> #
    simulations run. This is passed to all the major components of MCTS, which
    in turn can populate them however they would like. Finally this is
    returned, and thus can be used for reporting and debugging like.

    Lastly ``info`` returned will contain "plan_runtime" measurement.

    :param stop_cond: the function that returns whether simulating should stop
    :param tree_constructor: constructor the tree
    :param leaf_select: the method for selecting leaf nodes
    :param expand_and_eval: the leaf evaluation method
    :param backprop: the method for updating the statistics in the visited nodes
    :param action_select: the method for picking an action given root node
    :param horizon: the length of the problem
    :param belief: the current belief (over the state) at the root node
    :return: the preferred action and run time information (e.g. # simulations)
    """
    assert horizon > 0, f"Horizon ({horizon}) must be positive"

    info: Info = initiate_info()

    root_node = tree_constructor(belief, info)

    t = timer()

    while not stop_cond(info):

        state = belief()

        leaf, state, obs, terminal_flag, selection_output = leaf_select(
            state, root_node, info
        )

        assert (
            info["leaf_depth"] <= horizon
        ), f"Somehow the tree depth ({info['leaf_depth']}) exceeds the horizon ({horizon})"

        if not terminal_flag and info["leaf_depth"] < horizon:
            evaluation = expand_and_eval(leaf, state, obs, info)
        else:
            evaluation = None

        backprop(leaf, selection_output, evaluation, info)

        info["iteration"] += 1

    info["plan_runtime"] = timer() - t
    info["tree_root_stats"] = root_node.child_stats

    return action_select(root_node.child_stats, info), info


def deterministic_tree_search(
    stop_cond: StopCondition,
    tree_constructor: DeterministicTreeConstructor,
    leaf_select: DeterministicLeafSelection,
    expand_and_evaluate: DeterministicNodeExpansion,
    backprop: DeterministicBackPropagation,
    action_select: ActionSelection,
    history_representation: Any,
) -> Tuple[Action, Info]:
    """General deterministic tree search, defined by its components

    This search will simulate until ``stop_cond`` returns ``False``, where each
    simulation:

    #. Selects a leaf (action) node through ``leaf_select``
    #. Expands and evaluates the leaf node through ``expand_and_evaluate``
    #. Back propagates and updates node values through ``backprop``

    After spending the simulation budget, according to ``stop_cond`` it picks an
    given the statistics stored in the root node through ``action_select``.

    ``tree_constructor`` allows for custom ways of initiating the tree

    During run time will maintain information
    :class`~online-pomdp-planning.types.Info`, with "iteration" -> #
    simulations run. This is passed to all the major components of MCTS, which
    in turn can populate them however they would like. Finally this is
    returned, and thus can be used for reporting and debugging like.

    Lastly ``info`` returned will contain 'plan_runtime' measurement.

    :param stop_cond: the function that returns whether simulating should stop
    :param tree_constructor: constructor the tree
    :param leaf_select: the method for selecting leaf nodes
    :param expand_and_evaluate: the leaf expansion and evaluation method
    :param backprop: the method for updating the statistics in the visited nodes
    :param action_select: the method for picking an action given root node
    :param history_representation: whatever the input to ``tree_constructor``
    :return: the preferred action and run time information (e.g. # simulations)
    """

    info: Info = initiate_info()

    root_node = tree_constructor(history_representation, info)

    t = timer()

    while not stop_cond(info):

        leaf, selection_output = leaf_select(root_node, info)
        evaluation = expand_and_evaluate(leaf, info)

        backprop(leaf, selection_output, evaluation, info)

        info["iteration"] += 1

    info["plan_runtime"] = timer() - t

    return action_select(root_node.child_stats, info), info


def create_POUCT(
    actions: Sequence[Action],
    sim: Simulator,
    num_sims: int,
    init_stats: Optional[ActionStats] = None,
    leaf_eval: Optional[ExpandAndEvaluate] = None,
    ucb_constant: float = 1,
    horizon: int = 100,
    rollout_depth: int = 100,
    max_tree_depth: int = 100,
    discount_factor: float = 0.95,
    progress_bar: bool = False,
) -> Planner:
    """Creates PO-UCT given the available actions and a simulator

    Returns an instance of :func:`mcts` where the components have been
    filled in.

    Note::

        The ``horizon`` is *not updated* over time. This means that whenever
        the resulting planner is called, it assumes the current time step is 0.
        If you want MCTS to honor different timesteps, then call this function
        for every time step with an updated value for ``horizon``.

    There are multiple 'depths' to be set in POUCT. In particular there is the
    ``rollout_depth``, which specifies how many timesteps the random policy
    iterates in order to evaluate a leaf. Second the ``max_tree_depth`` is the
    maximum depth that the tree is allowed to grow. Lastly, the ``horizon`` is
    the actual length of the problem, and is an upperbound on both. This means,
    for example, that even if the ``rollout_depth`` is 3, if the horizon is 5
    then the random policy will only step once in order to evaluate it a node
    at depth 4, and that the tree will not grow past the ``horizon`` no matter
    the value of ``max_tree_depth``.

    :param actions: all the actions available to the agent
    :param sim: a simulator of the environment
    :param num_sims: number of simulations to run
    :param init_stats: how to initialize node statistics, defaults to None which sets Q and n to 0
    :param leaf_eval: the evaluation of leaves, defaults to ``None``, which assumes a random expand_and_rollout
    :param ucb_constant: exploration constant used in UCB, defaults to 1
    :param horizon: horizon of the problem (number of time steps), defaults to 100
    :param rollout_depth: the depth a expand_and_rollout will go up to, defaults to 100
    :param max_tree_depth: the depth the tree is allowed to grow to, defaults to 100
    :param discount_factor: the discount factor of the environment, defaults to 0.95
    :param progress_bar: flag to output a progress bar, defaults to False
    :return: MCTS with planner signature (given num sims)
    """
    assert num_sims > 0 and max_tree_depth > 0 and horizon > 0

    max_tree_depth = min(max_tree_depth, horizon)
    action_list = list(actions)

    if init_stats is None:
        init_stats = {a: {"qval": 0, "n": 0} for a in action_list}

    def generate_stats() -> ActionStats:
        """Returns initial action statistics

        This is to avoid giving _the same_ dictionary to various parts of the
        MCTS tree, accidentally sharing statistics where they should not

        :return: copy of ``init_stats``
        """
        assert init_stats is not None
        return {a: dict(stats) for a, stats in init_stats.items()}

    # stop condition: keep track of `pbar` if `progress_bar` is set
    pbar = no_stop
    if progress_bar:
        pbar = ProgressBar(num_sims)
    real_stop_cond = partial(has_simulated_n_times, num_sims)

    def stop_condition(info: Info) -> bool:
        return real_stop_cond(info) or pbar(info)

    def tree_constructor(belief, info):
        return create_root_node_with_child_for_all_actions(
            belief, info, generate_stats()
        )

    # defaults
    if not leaf_eval:
        assert rollout_depth > 0

        def expansion_strategy(o, action_node, info):
            return expand_node_with_all_actions(generate_stats(), o, action_node, info)

        def rollout_evaluation(leaf: ActionNode, s: State, o: Observation, info: Info):
            """Evaluates a leaf (:class:`ExpandAndEvaluate`) through random expand_and_rollout"""
            depth = min(rollout_depth, horizon - info["leaf_depth"])

            def policy(s, o):
                return random_policy(action_list, s, o)

            return expand_and_rollout(
                expansion_strategy,
                policy,
                sim,
                depth,
                discount_factor,
                leaf,
                s,
                o,
                info,
            )

        leaf_eval = rollout_evaluation

    node_scoring_method = partial(ucb_scores, ucb_constant=ucb_constant)
    leaf_select = partial(
        select_leaf_by_max_scores, sim, node_scoring_method, max_tree_depth
    )
    backprop = partial(
        backprop_running_q, discount_factor=discount_factor, backup_operator=mc_backup
    )
    action_select = max_q_action_selector

    return partial(
        mcts,
        stop_condition,
        tree_constructor,
        leaf_select,
        leaf_eval,
        backprop,
        action_select,
        horizon,
    )


def create_POUCT_with_model(
    actions: Sequence[Action],
    sim: Simulator,
    num_sims: int,
    leaf_eval_model: Callable[
        [Optional[ActionNode], State, Optional[Observation], Info],
        Tuple[float, ActionStats],
    ],
    ucb_constant: float = 1,
    horizon: int = 100,
    max_tree_depth: int = 100,
    discount_factor: float = 0.95,
    progress_bar: bool = False,
) -> Planner:
    """Creates PO-UCT given the available actions and a simulator

    Returns an instance of :func:`mcts` where the components have been
    filled in.

    In particular, it uses the ``leaf_eval_model`` to evaluate *and* get a
    prior whenever a node is expanded. Additionally, at the root creation a
    hundred states will be sampled to generate an (average) prior over the root
    action nodes.

    :param actions: all the actions available to the agent
    :param sim: a simulator of the environment
    :param num_sims: number of simulations to run
    :param leaf_eval_model: the state-based evaluation model
    :param ucb_constant: exploration constant used in UCB, defaults to 1
    :param horizon: the length of the problem
    :param max_tree_depth: the depth the tree is allowed to grow to, defaults to 100
    :param discount_factor: the discount factor of the environment, defaults to 0.95
    :param progress_bar: flag to output a progress bar, defaults to False
    :return: MCTS with planner signature (given num sims)
    """
    assert num_sims > 0 and max_tree_depth > 0 and horizon > 0

    action_list = list(actions)

    # stop condition: keep track of `pbar` if `progress_bar` is set
    pbar = no_stop
    if progress_bar:
        pbar = ProgressBar(num_sims)
    real_stop_cond = partial(has_simulated_n_times, num_sims)

    def stop_condition(info: Info) -> bool:
        return real_stop_cond(info) or pbar(info)

    def init_stats(a):
        return {"qval": 0, "prior": 1.0 / len(action_list), "n": 1}

    def tree_constructor(belief: Belief, info: Info) -> ObservationNode:
        """Custom-made tree constructor"""

        root = create_root_node_with_child_for_all_actions(
            belief, info, {a: init_stats(a) for a in action_list}
        )

        return root

    node_scoring_method = partial(alphazero_scores, ucb_constant=ucb_constant)

    leaf_select = partial(
        select_leaf_by_max_scores, sim, node_scoring_method, max_tree_depth
    )
    leaf_eval = partial(expand_and_evaluate_with_model, model=leaf_eval_model)
    backprop = partial(
        backprop_running_q, discount_factor=discount_factor, backup_operator=mc_backup
    )
    action_select = max_q_action_selector

    return partial(
        mcts,
        stop_condition,
        tree_constructor,
        leaf_select,
        leaf_eval,
        backprop,
        action_select,
        horizon,
    )


class MuzeroInferenceOutput(NamedTuple):
    """The output of calls to muzero's models"""

    value: float
    """The estimate value of the input (latent state/history)"""
    reward: float
    """The reward of a particular transition, seems to be ignored in initial inference"""
    policy: Dict[Action, float]
    """a => probability mapping"""
    latent_state: State
    """latent representation of the history/state"""


def create_muzero(
    initial_inference: Callable[[Any], MuzeroInferenceOutput],
    recurrent_inference: Callable[[State, Action], MuzeroInferenceOutput],
    num_sims: int,
    c1: float = 0.75,
    c2: float = 10,
    discount_factor: float = 0.95,
    root_noise_dirichlet_alpha: float = 10,
    root_noise_exploration_fraction: float = 0.2,
) -> Callable[[Any], Tuple[Action, Info]]:
    """Creates a muzero planner

    Runs :func:`deterministic_tree_search` like Muzero does. Initially maps
    some history representation to a latent state through ``initial_inference``,
    then continues running the tree search by picking nodes through statistics
    in nodes and expanding nodes with ``recurrent_inference``.

    Note that the 'type' of muzero is not well defined in this repository. The
    input is not a :class:`online_pomdp_planning.types.Belief`, but unclear
    what it is.

    NOTE::

        This planner is not tested and ran much, rather a prototype. Use at
        your own risk

    :param initial_inference: the initial mapping and inference over history
    :param recurrent_inference: the recurrent/dynamics/evaluation of tree expansion
    :param num_sims: number of tree expansions/simulations done
    :param c1: first exploration parameter used in mu-zero's UCB scoring method, defaults to 0.75
    :param c2: second exploration parameter used in mu-zero's UCB scoring method, defaults to 10
    :param discount_factor: the discount factor of the environment, defaults to 0.95
    :param root_noise_dirichlet_alpha: variance of (Dirichlet) root prior noise, default to 10
    :param root_noise_exploration_fraction: fraction of noise added to root prior, default to .2
    """
    stop_cond = partial(has_simulated_n_times, num_sims)

    def tree_constructor(history_representation: Any, info):
        inference = initial_inference(history_representation)
        return create_muzero_root(
            inference.latent_state,
            info,
            inference.reward,
            inference.policy,
            root_noise_dirichlet_alpha,
            root_noise_exploration_fraction,
        )

    # must use UCB with bounds, and deterministic nodes and everything
    # basically `muzero_pseudocode.py:468-471`
    node_scoring_method = partial(muzero_scores, c1=c1, c2=c2)
    leaf_select = partial(select_deterministc_leaf_by_max_scores, node_scoring_method)

    expand_and_evaluate = partial(muzero_expand_node, inference=recurrent_inference)
    action_select = max_visits_action_selector
    backprop = partial(deterministic_qval_backpropagation, discount_factor)

    return partial(
        deterministic_tree_search,
        stop_cond,
        tree_constructor,
        leaf_select,
        expand_and_evaluate,
        backprop,
        action_select,
    )
