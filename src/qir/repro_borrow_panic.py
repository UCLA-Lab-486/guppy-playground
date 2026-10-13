"""Minimal reproduction of Challenge A in guppy-to-qir.md, plus the §2.1
optimization comparison.

Chains N identical multi-controlled Pauli "lift" steps -- the same
`take -> with control -> put` shape as qvst._lift_mc -- and scales N to show
the array-borrow panic appearing once the default inliner's budget is exceeded,
then compares gate counts across optimization recipes.

Run from the repo root, e.g.:

    uv run src/qir/repro_borrow_panic.py

Expected shape of output (versions in guppy-to-qir.md):

    nsteps= 4  default                       OK   qis-gate-calls=16583
    nsteps= 4  minimal+Normalize(inline=All) OK   qis-gate-calls=16679
    nsteps= 4  +RemoveRedundancies           OK   qis-gate-calls=16583
    nsteps= 8  default                       FAIL Program may panic: Array already contains ...
    nsteps= 8  minimal+Normalize(inline=All) OK   qis-gate-calls=33351
    nsteps= 8  +RemoveRedundancies           OK   qis-gate-calls=33159
"""
import functools
import os
import re
import sys

# make src/ importable regardless of cwd (this file lives in src/qir/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import guppylang
from guppylang import guppy
from guppylang.std.builtins import array, result
from guppylang.std.quantum import qubit, measure_array, collect_measurements
from tket.passes import Normalize, PytketHugrPass
from tket.passes.inline_funcs import All
from pytket.passes import RemoveRedundancies
from hugr_qir.hugr_to_qir import hugr_to_qir
from hugr_qir.output import OutputFormat

from qubitization.util import multicontrol
from qubitization.block_encodings import pauli_term
from qubitization.qvst import _lift_mc, _chain

guppylang.enable_experimental_features()

N = 2            # system qubits (targets)
CN = 3           # control-register width
TOTAL = N + CN + 2
SYS = CN + 2

# one multi-controlled Pauli term, lifted onto qs like qvst's SELECT
term = pauli_term(N, [("z", 0), ("z", 1)])
mck = multicontrol(term, n_targets=N, n_controls=CN + 1, control_state=1)
step1 = _lift_mc(TOTAL, mck, 1, CN + 1, SYS, N)

RR = PytketHugrPass(RemoveRedundancies())


def build_circuit(nsteps):
    if nsteps == 1:
        return step1
    return functools.reduce(lambda f, g: _chain(TOTAL, f, g), [step1] * nsteps)


def count_gates(qir):
    return len(re.findall(r"call[^\n]*__quantum__qis__", qir))


for nsteps in (4, 8):
    circuit = build_circuit(nsteps)

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(TOTAL))
        circuit(qs)
        result("c", collect_measurements(measure_array(qs)))

    recipes = {
        "default": lambda: main.compile(),
        "minimal+Normalize(inline=All)": lambda: (
            main.with_minimal_opt()
            .with_optimization(Normalize(inline_funcs=All()))
            .compile()
        ),
        "+RemoveRedundancies": lambda: (
            main.with_minimal_opt()
            .with_optimization(Normalize(inline_funcs=All()))
            .with_optimization(RR)
            .compile()
        ),
    }
    for label, builder in recipes.items():
        try:
            qir = hugr_to_qir(builder(), output_format=OutputFormat.LLVM_IR)
            print(f"nsteps={nsteps:2d}  {label:30s} OK   qis-gate-calls={count_gates(qir)}")
        except Exception as e:
            print(f"nsteps={nsteps:2d}  {label:30s} FAIL {str(e).splitlines()[0][:50]}")
