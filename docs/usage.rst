=====
Usage
=====

Here we give an example on how to solve the canonical tiger problem. First we
define the environment:

.. literalinclude:: ../tests/test_running_planners.py
   :pyobject: Tiger

Then given some beliefs:

.. literalinclude:: ../tests/test_running_planners.py
   :pyobject: uniform_tiger_belief

.. literalinclude:: ../tests/test_running_planners.py
   :pyobject: tiger_left_belief

.. literalinclude:: ../tests/test_running_planners.py
   :pyobject: tiger_right_belief

Then this library solves for the particular beliefs::

    from online_pomdp_planning.mcts import create_POUCT

    n_sims = 2 * 16384
    ucb_constant = 100

    planner = create_POUCT(Tiger.actions(), Tiger.sim, n_sims, ucb_constant=ucb_constant)

    # action for uniform belief
    action, info = planner(uniform_tiger_belief)
    assert action == Tiger.H
    assert info["iteration"] == n_sims

    # action for left belief
    action, info = planner(tiger_left_belief)
    assert action == Tiger.L
    assert info["iteration"] == n_sims
