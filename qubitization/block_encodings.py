"""
This file contains circuits for the block encoded hamiltonians.
"""

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, h, measure, qubit, x


n = guppy.nat_var("n")

@guppy
def heis_chain(qubit_numbers: int) -:
    pass