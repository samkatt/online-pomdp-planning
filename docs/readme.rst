=====================
online-pomdp-planning
=====================

.. POMDPs

Partially observable Markov decision processes (POMDP
[kaelbling_planning_1998]_) is a mathematical framework for defining
reinforcement learning (RL) in environments with hidden state. To solve the RL
problem means to come up with a policy, a mapping from the past observations of
the environment to an action.

.. online planning

Online planning is the family of methods that assumes access to (a simulator
of) the dynamics and infers what action to take *during execution*. For this it
requires the belief, a probability distribution over the current state. The
planner takes a current belief of the current state of the environment and a
simulator, and spits out its favorite action.

.. [kaelbling_planning_1998] Kaelbling, Leslie Pack, Michael L. Littman, and
   Anthony R. Cassandra. “Planning and acting in partially observable
   stochastic domains.“ Artificial intelligence 101.1-2 (1998): 99-134.

Methods
-------

This library implements a set of these methods:

.. toctree::
   :maxdepth: 1

   methods/mcts
   methods/despot
   methods/f3s

.. provided in this package

Concretely, this package provides factory functions to construct
:class:`~online_pomdp_planning.types.Planner`. A planner is a function that
is called with a :class:`~online_pomdp_planning.types.Belief`, and returns a
:class:`~online_pomdp_planning.types.Action`.

.. automethod:: online_pomdp_planning.types.Planner.__call__
   :noindex:

Types
-----

I am unreasonably terrified of dynamic typed languages and have gone to
extremes to define as many as possible. Most of these are for internal use, but
you will come across some as a user of this library. Most of these types will
have no actual meaning, in particular:

.. autosummary::
   :nosignatures:

   online_pomdp_planning.types.Action
   online_pomdp_planning.types.Observation
   online_pomdp_planning.types.State

.. meaningless types: `Action`, `Observation` & `State`

Are domain specific and unimportant. All that is required is that the
:class:`~online_pomdp_planning.types.Action` and
:class:`~online_pomdp_planning.types.Observation` are `hashable`. The
:class:`~online_pomdp_planning.types.State` is not used by the library code
whatsoever.

.. `Belief` type

A notable exception is the :class:`~online_pomdp_planning.types.Belief`,
which is assumed to a callable that produces states. This represent that we
assume the belief is a way of sampling states.

.. automethod:: online_pomdp_planning.types.Belief.__call__
   :noindex:

