import numpy as np

from guppylang import guppy, comptime
from guppylang.std.builtins import array, result
from guppylang.std.quantum import cx, h, qubit, measure_array, collect_measurements
from tket.passes import Normalize, PytketHugrPass
from tket.passes.inline_funcs import All
from pytket.passes import RemoveRedundancies

from hugr_qir.hugr_to_qir import hugr_to_qir
from hugr_qir.output import OutputFormat

from qubitization.params import num_qubits, time_t, J, h
from qubitization.qvst import qvst_algo
from qubitization.block_encodings import qvst_heis_chain

# @guppy
# def main() -> None:
#     qs = array(qubit() for _ in range(2))
#     h(qs[0])
#     cx(qs[0], qs[1])
#     result("c", collect_measurements(measure_array(qs)))
# 
# hugr_module = main.compile()

circuit, total, control_numbers = qvst_algo(num_qubits, time_t, qvst_heis_chain)

@guppy
def entrypoint() -> None:
    qs = array(qubit() for _ in range(total))
    circuit(qs)
    result("c", collect_measurements(measure_array(qs)))

# The block-encoding steps borrow disjoint, comptime-constant sub-slices of the
# qubit array via array.take/.put. Those ops carry runtime borrow/bounds panic
# checks that H-Series rejects ("Runtime panics are unsupported on H-Series").
# Fully inlining every sub-circuit (inline_funcs=All) lets Normalize's
# squash_borrows prove the constant indices safe and delete the panic checks.
# The default opt level's inline heuristic doesn't inline aggressively enough.
#
# with_minimal_opt() clears the default pass list, which also includes pytket's
# RemoveRedundancies (quantum peephole: cancels adjacent redundant gates), so we
# re-add it after Normalize to keep the gate-level optimization the default
# pipeline would have done. This reproduces Default's gate count while staying
# panic-free (RemoveRedundancies runs after borrows are already squashed).
hugr_module = (
    entrypoint.with_minimal_opt()
    .with_optimization(Normalize(inline_funcs=All()))
    .with_optimization(PytketHugrPass(RemoveRedundancies()))
    .compile()
)

qir_llvm_text = hugr_to_qir(hugr_module, output_format=OutputFormat.LLVM_IR)

with open("qir.txt", "w") as f:
    print(qir_llvm_text, file=f)
