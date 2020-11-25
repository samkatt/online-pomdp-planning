===========================================
Partially observable upper confidence bound
===========================================

Monte-Carlo tree search, first implemented in POMCP [silver_monte-carlo_2010]_
implementation where the components are:

- leaf selection
    - action generation: upper-confidence bound for action selection
    - observation generation: simulator for observation generator
- leaf expansion:
    - add observation node with child for each action node
- leaf evaluation:
    - rollout (default is random)
- back propagation:
    - running Q-average
- action selector:
    - max Q

.. TODO: provide and describe constructor

.. [silver_monte-carlo_2010] Silver, David, and Joel Veness. "Monte-Carlo planning in
   large POMDPs." Advances in neural information processing systems. 2010.
