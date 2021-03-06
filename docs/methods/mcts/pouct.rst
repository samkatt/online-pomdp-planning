===========================================
Partially observable upper confidence bound
===========================================

Monte-Carlo tree search, first implemented in POMCP [silver_monte-carlo_2010]_
implementation where the components are:

- root node construct: :py:func:`~online_pomdp_planning.mcts.create_root_node_with_child_for_all_actions`
    - simply initiates a action node for each action
- stop condition: :py:func:`~online_pomdp_planning.mcts.has_simulated_n_times`
    - simulate exactly ``n`` times
- leaf selection :py:func:`~online_pomdp_planning.mcts.ucb_select_leaf`
    - action generation: upper-confidence bound for action selection
    - observation generation: simulator for observation generator
- leaf expansion :py:func:`~online_pomdp_planning.mcts.expand_node_with_all_actions`:
    - expand a new node with references for all provided actions
- leaf evaluation :py:func:`~online_pomdp_planning.mcts.rollout`:
    - rollout of some policy given a simulator (default is random)
- back propagation :py:func:`~online_pomdp_planning.mcts.backprop_running_q`:
    - running Q-average
- action selector :py:func:`~online_pomdp_planning.mcts.max_q_action_selector`:
    - picks the action with max q value

We provide a relatively easy way of constructing PO-UCT for your usage through

.. autofunction:: online_pomdp_planning.mcts.create_POUCT
   :noindex:

.. [silver_monte-carlo_2010] Silver, David, and Joel Veness. "Monte-Carlo
   planning in large POMDPs.“ Advances in neural information processing
   systems. 2010.
