"""Options extension to MCTS"""

from functools import partial
from typing import Any, List, NamedTuple, Protocol, Sequence, Tuple

from online_pomdp_planning.mcts import (
    ActionNode,
    ActionScoringMethod,
    ActionStats,
    ObservationNode,
    ProgressBar,
    backprop_running_q,
    create_root_node_with_child_for_all_actions,
    expand_node_with_all_actions,
    has_simulated_n_times,
    max_q_action_selector,
    mc_backup,
    mcts,
    no_stop,
    select_action,
    ucb_scores,
)
from online_pomdp_planning.types import (
    Action,
    Info,
    Observation,
    Planner,
    Simulator,
    State,
)


class StopCondition(Protocol):
    """Represents the stop condition in :class:`Option`

    .. automethod:: __call__
    """

    def __call__(self, o: Observation) -> bool:
        """The signature of a stop condition

        .. todo::

            - Consider making it dependent on the history

        :param o: the last observation to help decide whether to terminate
        :return: whether or not to terminate (can be stochastic)
        """
        raise NotImplementedError()


class OptionPolicy(Protocol):
    """Represents the policy of an :class:`Option`

    .. automethod:: __call__
    """

    def __call__(self, o: Observation) -> Action:
        """The signature of a policy in an option, simply reactive to last observation

        :param o: the (last) observation to base the action on
        :return: which action to take given `o`
        """
        raise NotImplementedError()


class Option(NamedTuple):
    """Represents an option in RL literature

    An option typical allows for temporal abstraction by substituting for
    primitive actions. An option has a policy that decides on primitive
    actions, and some (probabilistic) stopping condition.

    This class is simply a container for both.
    """

    policy: OptionPolicy
    stop_condition: Any


def action_to_option(a: Action, cond: StopCondition) -> Option:
    """Creates an :class:`Option` whose policy is always doing `a` and stopping when `cond`"""
    return Option(lambda o: a, cond)


def apply_option(
    option: Option, state: State, obs: Observation, sim: Simulator
) -> Tuple[State, Observation, List[float], bool]:
    # at least call our option once
    s, o, r, t = sim(state, option.policy(obs))

    rewards = [r]
    # TODO: clean up, must be a better way? Whale operator?
    while not t and not option.stop_condition(s, o):
        s, o, r, t = sim(s, option.policy(o))
        rewards.append(r)

    return s, o, rewards, t


def select_leaf_with_options(
    sim: Simulator,
    scoring_method: ActionScoringMethod,
    discount_factor: float,
    state: State,
    node: ObservationNode,
    info: Info,
) -> Tuple[ActionNode, State, Observation, bool, Any]:
    """Selects a leaf in the option-observation tree

    Given a simulator and (root?) node, this function will select the next leaf
    to evaluate/expand etc. Picks options according to
    :func:`~online_pomdp_planning.mcts.select_action` --- with
    ``scoring_method`` --- and calls :func:`apply_option` to simulate
    interactions.

    Implements :class:`~online_pomdp_planning.mcts.LeafSelection` when provided
    with `sim` and `scoring_method`.

    See :func:`~online_pomdp_planning.mcts.unified_ucb_scores` for ways of
    constructing ``scoring_method``

    NOTE::

        Assumes 'actions' in the statistics stored in `node` of tree are options.

        Assumes "ucb_num_terminal_sims", "ucb_tree_depth" and "leaf_depth" to
        be entries in `info`

        Tracks the tree depth, maintains a running statistic on it in `info`
        ("ucb_tree_depth") and sets "leaf_depth" in `info`.

    :param sim: a POMDP simulator
    :param scoring_method: function that, given action stats, returns their scores
    :param discount_factor: necessary to compute the (discounted) reward of branches
    :param max_depth: max length of the tree to go down
    :param state: the root state
    :param node: the root node
    :param info: run time information (ignored)
    :return: leaf node, state, observation, terminal flag and list of rewards
    """
    assert 0 < discount_factor <= 1
    list_of_discounted_rewards: List[float] = []

    # A tree traversal step corresponds to following an option
    # So the `time_step` _does not correspond_ to `depth`
    depth = 0
    time_step = 0

    # traverse the tree to leaf or terminal transition
    while True:

        option = select_action(node.child_stats, info, scoring_method)
        assert isinstance(option, Option)

        state, obs, rewards, terminal_flag = apply_option(
            option, state, node.observation, sim
        )

        # We compute the discounted rewards (w.r.t\ time steps)
        # After the call of this function no-one can reproduce the time steps
        # Hence we compute it here, any other application of discount is wrong
        list_of_discounted_rewards.append(
            sum(r * discount_factor ** (time_step + k) for k, r in enumerate(rewards))
        )

        # Again, one step in the tree might relate to many time steps
        depth += 1
        time_step += len(rewards)

        if terminal_flag:
            break

        try:
            # NOTE: here we branch off of `Option` and (last!) `Observation`
            node = node.action_node(option).observation_node(obs)
        except KeyError:
            # action node is a leaf
            break

    # info tracking number of terminal simulations
    if terminal_flag:
        info["ucb_num_terminal_sims"] += 1

    # info tracking tree depth
    info["ucb_tree_depth"].add(depth)
    info["leaf_depth"] = depth
    return (
        node.action_node(option),
        state,
        obs,
        terminal_flag,
        list_of_discounted_rewards,
    )


def create_POUCT_with_options(
    options: Sequence[Option],
    sim: Simulator,
    num_sims: int,
    ucb_constant: float = 1,
    discount_factor: float = 0.95,
    progress_bar: bool = False,
) -> Planner:
    """Call this to create a planner for options

    Assumes _you_ give the `options` - see :func:`action_to_option` for
    inspiration. It basically applies :func:`~online_pomdp_planning.mcts.mcts`
    with `options` as actions. The key difference is that one tree 'step'
    (branching) corresponds to potentially many `sim` steps.

    It will run `num_sims` simulations using
    :func:`~online_pomdp_planning.mcts.ucb_scores` to pick 'options' and then
    :func:`apply_option` for getting a list of discounted rewards and
    observation.

    The returned MCTS currently has standard implementation choices for things
    as back-propagation (running q average), tree constructor (initialization
    statistics), or action selection (max-q)

    .. todo
        - The typical `horizon` (or `max_tree_depth`) argument has been
          removed. This is mainly because it is kinda complicated in which
          places exactly this should be checked for. So for now future work.

        - Add the ability to evaluate leafs, the initialization of (option)
          statistics is not informed, returning an expected value of 0 and no
          preference for the options.

    :param options: all the options available to the agent
    :param sim: a simulator of the environment
    :param num_sims: number of simulations to run
    :param ucb_constant: exploration constant used in UCB, defaults to 1
    :param discount_factor: the discount factor of the environment, defaults to 0.95
    :param progress_bar: flag to output a progress bar, defaults to False
    :return: MCTS with planner signature
    """
    assert num_sims > 0

    option_list = list(options)

    def generate_stats() -> ActionStats:
        return {o: {"qval": 0, "n": 0} for o in option_list}

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

    # Currently our 'evaluation' simply returns 0.
    # This is future work
    def expand_and_evaluate(
        leaf: ActionNode, s: State, o: Observation, info: Info
    ) -> float:
        expand_node_with_all_actions(generate_stats(), o, leaf, info)
        return 0

    node_scoring_method = partial(ucb_scores, ucb_constant=ucb_constant)
    leaf_select = partial(
        select_leaf_with_options, sim, node_scoring_method, discount_factor
    )

    # We trick back-propagation here.
    # It does not how many time steps each step took due to options,
    # and thus cannot compute the discounted reward itself.
    # It is important that the rewards returned in `leaf_select` are
    # already discounted, and apply a 1.0 discount here, for accuracy
    backprop = partial(
        backprop_running_q, discount_factor=1.0, backup_operator=mc_backup
    )
    action_select = max_q_action_selector

    return partial(
        mcts,
        stop_condition,
        tree_constructor,
        leaf_select,
        expand_and_evaluate,
        backprop,
        action_select,
        float("inf"),  # infinite horizon, since it does not work well
    )
