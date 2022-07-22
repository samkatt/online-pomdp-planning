"""Extension of planning in POMDPs: options

Options are 'macro-actions' or 'temporal abstractions' that last over multiple
time steps. This is useful for longer term planning, given that the options are
good. An option typically consists of a 'policy', and a 'terminal condition'.
The details of such implementations are often different per implementation.

This module attempts to include options, as general as possible, into existing
planners.

"""
