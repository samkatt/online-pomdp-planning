=======================
Monte-Carlo tree search
=======================

.. MCTS

Monte-Carlo tree search (MCTS [browne_survey_2012]_) in POMDPs incrementally
builds a look-ahead tree of interactions with the environment. This is done
through simulations, where each simulation travels through the tree and expands
when it reaches a leaf. In POMDPs the tree branches on actions and observations
[silver_monte-carlo_2010]_.

Short description
=================

A simulation starts by sampling a state from the belief, and consists of 4
steps:

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

Implementation details
======================

This library implements MCTS as a combination of:

#. A `Tree Policy`_
#. An `Observation generator`_
#. An `Expansion`_ method
#. An `Evaluation`_ method
#. A `Back propagation`_ method
#. An `Action selector`_ method given the root statistics

The idea is that typical implementations of these components are given, but
that it is easy to create your own to adapt MCTS to your own liking (e.g.\
provide your own evaluation method).

Tree policy
-----------

.. english description

The tree policy is a function that takes in the statistics of a node and
returns an action. This mapping is being used to pick actions while traversing
a tree. A popular choice is upper confidence bound (UCB)
[auer_finite-time_2002]_.

.. TODO: signature and list of functions

Observation generator
---------------------

The observation generator generates the observations *while traversing the
tree*. Typically a simulator is used to do so, but in domains with a large
(e.g.\  continuous) observation space methods such as progressive widening to
limit the branching factor on observations [sunberg_online_2018]_
[couetoux2011continuous]_.

.. TODO: signature and list of functions

Expansion
---------

The expansion method decides how to grow the tree upon reaching a leaf. The
most common approach is to add a node that represents the simulated
(action-observation) 'history' with some initial statistics.

.. TODO: signature and list of functions

Evaluation
----------

The evaluation of a leaf node gives an estimated return (value) of a node,
called when a leaf is reached. A common implementation is a (random)
interaction with the environment. More effective implementations can be
domain-specific policies or evaluation models (e.g.\  neural networks trained
to evaluate a history).

.. TODO: signature and list of functions

Back propagation
----------------

The back propagation method implements how a visited node should be updated and
takes in the node and the return of the simulation (after picking this node).
It is expected that the implementation updates the node statistics one way or
another. The most common implementation tracks the number of node visits
and the average return.

.. TODO: signature and list of functions

Action selector
---------------

Lastly, the action selector decides how to pick the action given the statistics
in the root node. A common method is to pick the action with highest associated
average return.

.. TODO: signature and list of functions

.. References

.. [browne_survey_2012] Browne, Cameron B., et al. "A survey of monte carlo
   tree search methods." IEEE Transactions on Computational Intelligence and AI
   in games 4.1 (2012): 1-43.

.. [silver_monte-carlo_2010] Silver, David, and Joel Veness. "Monte-Carlo planning in
   large POMDPs." Advances in neural information processing systems. 2010.

.. [auer_finite-time_2002] Auer, Peter, Nicolo Cesa-Bianchi, and Paul Fischer.
   "Finite-time analysis of the multiarmed bandit problem." Machine learning
   47.2-3 (2002): 235-256.

.. [sunberg_online_2018] Sunberg, Zachary, and Mykel Kochenderfer. "Online
   algorithms for POMDPs with continuous state, action, and observation
   spaces." arXiv preprint arXiv:1709.06196 (2017).

.. [couetoux2011continuous] Couëtoux, Adrien, et al. "Continuous upper
   confidence trees." International Conference on Learning and Intelligent
   Optimization. Springer, Berlin, Heidelberg, 2011.
