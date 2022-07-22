===========================================
Partially observable upper confidence bound
===========================================

Monte-Carlo tree search, first implemented in POMCP [silver_monte-carlo_2010]_
implementation where the components are:

- root node construct: :func:`~online_pomdp_planning.mcts.create_root_node_with_child_for_all_actions`
    - simply initiates a action node for each action
- stop condition: :func:`~online_pomdp_planning.mcts.has_simulated_n_times`
    - simulate exactly ``n`` times
- leaf selection :func:`~online_pomdp_planning.mcts.select_leaf_by_max_scores`
    - pick nodes according to their upper-confidence score
    - observation generation: simulator for observation generator
- leaf expansion :func:`~online_pomdp_planning.mcts.expand_node_with_all_actions`:
    - expand a new node with references for all provided actions
- leaf evaluation :func:`~online_pomdp_planning.mcts.rollout`:
    - rollout of some policy given a simulator (default is random)
- back propagation :func:`~online_pomdp_planning.mcts.backprop_running_q`:
    - running Q-average
- action selector :func:`~online_pomdp_planning.mcts.max_q_action_selector`:
    - picks the action with max q value

We provide a relatively easy way of constructing PO-UCT for your usage through

.. autofunction:: online_pomdp_planning.mcts.create_POUCT
   :noindex:

Options
-------

.. code design

In practice, planning with options requires to branch off of options rather
than primitive actions. Such a branch is created by running the option's policy
against the simulator, and then branch again off of the last observation. The
reward associated with this must be properly discounted (w.r.t the time), and
dealing with the horizon is slightly more complicated. 

.. options implementation

An :class:`~online_pomdp_planning.options.mcts.Option` is simply a container of
a :class:`~online_pomdp_planning.options.mcts.OptionPolicy` and
:class:`~online_pomdp_planning.options.mcts.StopCondition`.

.. option example

For example, to create a stop condition that will randomly terminate, consider

.. literalinclude:: ../../../tests/options/test_running_option_planners.py
   :pyobject: random_stop_condition

Then, for example, create an option that continues to do some primitive
:class:`~online_pomdp_planning.types.Action`:

.. literalinclude:: ../../../online_pomdp_planning/options/mcts.py
   :pyobject: action_to_option

Then, to create a planner using these, consider

.. autofunction:: online_pomdp_planning.options.mcts.create_POUCT_with_options
   :noindex:

.. [silver_monte-carlo_2010] Silver, David, and Joel Veness. "Monte-Carlo
   planning in large POMDPs.“ Advances in neural information processing
   systems. 2010.
