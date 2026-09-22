"""
Smoke test for qvst_algo: run on the all-zero state and check the
ancilla-zero block approximates exp(-i H t / norm) |0...0>.
"""

import numpy as np

from guppylang import guppy, comptime
from guppylang.emulator import EmulatorBuilder
from guppylang.std.builtins import array
from guppylang.std.debug import state_result
from guppylang.std.quantum import discard_array, qubit
from tket.passes import ModifierResolverPass

from qubitization.block_encodings import qvst_heis_chain, qvst_ising_chain
from qubitization.qvst import qvst_algo


def run_circuit(circ, total):
    @guppy
    def sim() -> None:
        qs = array(qubit() for _ in range(comptime(total)))
        circ(qs)
        state_result("out", qs)
        discard_array(qs)

    pack = sim.compile()
    ModifierResolverPass().run(pack.modules[0])
    emu = EmulatorBuilder().build(pack, n_qubits=total)
    return emu.run().partial_state_dicts()[0]["out"].state_distribution()[0].state


def smoke(name, builder, expect_cn):
    num_qubits = 2
    time_t = 0.0070711

    circ, total, cn = qvst_algo(num_qubits, time_t, builder)
    print(f"{name}: CN={cn}, total={total}")
    assert cn == expect_cn, f"expected CN={expect_cn}, got {cn}"
    assert total == num_qubits + cn + 2

    state = np.asarray(run_circuit(circ, total))
    print(f"{name}: state norm = {np.linalg.norm(state):.6f}")
    assert abs(np.linalg.norm(state) - 1) < 1e-6

    # Ancilla-zero block is indices 0 .. 2^N - 1 (big-endian, ancillas high).
    dim = 2**num_qubits
    block = state[:dim]
    block_weight = np.linalg.norm(block) ** 2
    overlap = abs(block[0]) / np.linalg.norm(block)
    print(f"{name}: block state (indices 0..{dim - 1}) = {np.round(block, 5)}")
    print(f"{name}: weight on ancilla-zero block = {block_weight:.6f}")
    print(f"{name}: |<e_0, block>| / ||block|| = {overlap:.6f}")
    assert block_weight > 0.99, f"ancilla-zero weight too low: {block_weight}"
    assert overlap > 0.99, f"overlap with |00> too low: {overlap}"
    print(f"{name}: OK\n")


if __name__ == "__main__":
    smoke("heis", qvst_heis_chain, expect_cn=3)
    smoke("ising", qvst_ising_chain, expect_cn=2)
    print("smoke test passed")
