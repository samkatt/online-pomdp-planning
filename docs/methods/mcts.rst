=======================
Monte-Carlo tree search
=======================

.. MCTS

Monte-Carlo tree search (MCTS [browne_survey_2012]_) in POMDPs incrementally
builds a look-ahead tree of interactions with the environment. This is done
through simulations, where each simulation travels through the tree and expands
when it reaches a leaf. In POMDPs the tree branches on actions and observations.

Short description
=================

First a root node is created, then the algorithm proceeds with simulations.  A
simulation starts by sampling a state from the belief, and consists of 4 steps:

#. Select a leaf node by traversing through the tree
#. Expand the chosen leaf node
#. Evaluate the leaf node
#. Update the values in the visited nodes

.. figure:: figures/MCTS-steps.svg
   :align: center

   Visualisation of MCTS, borrowed from `wikipedia
   <https://en.wikipedia.org/wiki/Monte_Carlo_tree_search>`_

At the end of the simulation budget the statistics in the root of the tree are
used to pick the action. After executing the action and an observation is
perceived the belief updated and we repeat.

.. toctree::
   :maxdepth: 1

   mcts/pouct

Implementation details
======================

This library implements MCTS as a combination of:

#. `Tree construction`_
#. `Stop condition`_
#. `Leaf selection`_
#. `Leaf expansion`_
#. `Leaf Evaluation`_
#. `Back propagation`_
#. `Action selector`_

The idea is that typical implementations of these components are given, but
that it is easy to create your own to adapt MCTS to your own liking (e.g.\
provide your own evaluation method).

Tree construction
-----------------

This function simply returns the root node of an initial 'tree'. A fair share
of domain knowledge can be used here, but a typical initiation simply creates
the first set of action nodes, one for each action.

.. autofunction:: online_pomdp_planning.mcts.TreeConstructor.__call__
   :noindex:


Provided implementations:

    - :func:`~online_pomdp_planning.mcts.create_root_node_with_child_for_all_actions`


Stop condition
--------------

This function determines when MCTS stops. Typically this can either be a timer,
or a desired number of simulations.

.. autofunction:: online_pomdp_planning.mcts.StopCondition.__call__
   :noindex:

Provided implementations:

    - :func:`~online_pomdp_planning.mcts.has_simulated_n_times`
    - :func:`~online_pomdp_planning.mcts.ProgressBar`


Leaf selection
--------------

The tree policy is a function that takes in the statistics of a node and
returns an action. This mapping is being used to pick actions while traversing
a tree:

.. automethod:: online_pomdp_planning.mcts.LeafSelection.__call__
   :noindex:

A popular choice is upper confidence bound (UCB) [auer_finite-time_2002]_.

Provided implementations:

    - :func:`~online_pomdp_planning.mcts.select_leaf_by_max_scores`

Leaf expansion
--------------

The expansion method decides how to grow the tree upon reaching a leaf. The
most common approach is to add a node that represents the simulated
(action-observation) 'history' with some initial statistics:

.. automethod:: online_pomdp_planning.mcts.Expansion.__call__
   :noindex:

Provided implementations:

    - :func:`~online_pomdp_planning.mcts.expand_node_with_all_actions`
      expand a new node with references for all provided actions

Leaf evaluation
---------------

The evaluation of a leaf node gives an estimated return (value) of a node,
called when a leaf is reached:

.. automethod:: online_pomdp_planning.mcts.Evaluation.__call__
   :noindex:

A common implementation is a (random) interaction with the environment. More
effective implementations can be domain-specific policies or evaluation models
(e.g.\  neural networks trained to evaluate a history).

Provided implementations:

    - :func:`~online_pomdp_planning.mcts.rollout`
    - :func:`~online_pomdp_planning.mcts.state_based_model_evaluation`

Back propagation
----------------

The back propagation method implements how a visited node should be updated and
takes in the node and the return of the simulation (after picking this node).
It is expected that the implementation updates the node statistics one way or
another.

.. automethod:: online_pomdp_planning.mcts.BackPropagation.__call__
   :noindex:

The most common implementation tracks the number of node visits and the average
return.

Provided implementations:

    - :func:`~online_pomdp_planning.mcts.backprop_running_q`
    - :func:`~online_pomdp_planning.mcts.associate_prior_with_nodes_of_child`

Action selector
---------------

Lastly, the action selector decides how to pick the action given the statistics
in the root node.

.. automethod:: online_pomdp_planning.mcts.ActionSelection.__call__
   :noindex:

Provided implementations:

    - :func:`~online_pomdp_planning.mcts.max_q_action_selector` picks the action with max q value
    - :func:`~online_pomdp_planning.mcts.max_visits_action_selector` picks the action with max visits
    - :func:`~online_pomdp_planning.mcts.visit_prob_action_selector` sample the action according to visitation

A common method is to pick the action with highest associated average return.

Discrete search
===============

This package provides a slight variant to MCTS discussed so far: deterministic
search. This approach assumes that the result of each action is deterministic,
meaning it has one outcome ("observation" or "state"). The resulting algorithm
is slightly different, and is provided through deterministic variants of the
previous interfaces (note expansion and evaluation is merged here):

    - :class:`~online_pomdp_planning.mcts.DeterministicTreeConstructor`, with implementation(s):
        - :class:`~online_pomdp_planning.mcts.create_muzero_root`
    - :class:`~online_pomdp_planning.mcts.DeterministicLeafSelection`, with implementation(s):
        - :class:`~online_pomdp_planning.mcts.select_deterministc_leaf_by_max_scores`
            - with scoring method :func:`~online_pomdp_planning.mcts.muzero_ucb_scores`
    - :class:`~online_pomdp_planning.mcts.DeterministicNodeExpansion`, with implementation(s):
        - :func:`~online_pomdp_planning.mcts.muzero_expand_node`
    - :func:`~online_pomdp_planning.mcts.DeterministicBackPropagation`, with implementation(s):
        - :func:`~online_pomdp_planning.mcts.deterministic_qval_backpropagation`

Currently the (only) implemented discrete tree search is Muzero
[schrittwieser2020mastering]_
(:func:`online_pomdp_planning.mcts.create_muzero`), given the correct methods
to do the initial and recurrent inference.

.. References

.. [browne_survey_2012] Browne, Cameron B., et al. "A survey of monte carlo
   tree search methods." IEEE Transactions on Computational Intelligence and AI
   in games 4.1 (2012): 1-43.

.. [auer_finite-time_2002] Auer, Peter, Nicolo Cesa-Bianchi, and Paul Fischer.
   "Finite-time analysis of the multiarmed bandit problem." Machine learning
   47.2-3 (2002): 235-256.

.. [schrittwieser2020mastering] Schrittwieser, J., Antonoglou, I., Hubert, T.,
   Simonyan, K., Sifre, L., Schmitt, S., ... & Silver, D. (2020). Mastering
   atari, go, chess and shogi by planning with a learned model. Nature,
   588(7839), 604-609.
