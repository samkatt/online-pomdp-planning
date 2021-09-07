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

.. [silver_monte-carlo_2010] Silver, David, and Joel Veness. "Monte-Carlo
   planning in large POMDPs.“ Advances in neural information processing
   systems. 2010.
