===========================================
Partially observable upper confidence bound
===========================================

Monte-Carlo tree search, first implemented in POMCP [silver_monte-carlo_2010]_
implementation where the components are:

- leaf selection :py:func:`~online_pomdp_planning.mcts.select_with_ucb`
    - action generation: upper-confidence bound for action selection
    - observation generation: simulator for observation generator
- leaf expansion :py:func:`~online_pomdp_planning.mcts.expand_node_with_all_actions`:
    - expand a new node with references for all provided actions
- leaf evaluation:
    - rollout (default is random)
- back propagation:
    - running Q-average
- action selector :py:func:`~online_pomdp_planning.mcts.pick_max_q`:
    - picks the action with max q value

.. todo::

    provide and describe constructor

.. [silver_monte-carlo_2010] Silver, David, and Joel Veness. "Monte-Carlo planning in
   large POMDPs." Advances in neural information processing systems. 2010.
