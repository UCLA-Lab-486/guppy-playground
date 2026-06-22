"""Steane [[7, 1, 3]] quantum error-correcting code in Guppy.

The Steane code encodes one *logical* qubit into seven *physical* (data) qubits
and can correct any single-qubit error.  This file implements the code in a
modular way: every logical operation is its own function that acts on a block of
seven physical qubits, so the logical circuit reads almost exactly like the bare
physical circuit in ``main.py``.

Conventions follow the paper ``fault-tolerant-execution.pdf`` (Sec. II), which
uses the qubit ordering of Refs. [20, 22, 44-46] and the [7, 4, 3] Hamming parity
check matrix

        h = [[0 0 0 1 1 1 1],
             [0 1 1 0 0 1 1],
             [1 0 1 0 1 0 1]].

The rows of ``h`` define both the X- and Z-type stabilizer generators, e.g.
``g_X1 = X3 X4 X5 X6``.  The logical operators are (Eq. 2)

        X_L  ~  X0 X1 X2          Z_L  ~  Z0 Z1 Z2.

The logical Clifford gates H, S and CNOT are *transversal* (Sec. II B):
``H_L = prod_j H_j``, ``S_L = prod_j Sdg_j``, ``CNOT_L = prod_j CNOT(j, j+7)``.
"""

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.lang import owned
from guppylang.std.quantum import cx, h, sdg, x, z, measure, measure_array, qubit
from guppylang.std.array import array

# ---------------------------------------------------------------------------
# Encoding: prepare a logical |0>
# ---------------------------------------------------------------------------
# |0_L> is the +1 eigenstate of every stabilizer and of Z_L.  We build it with
# the standard (non-fault-tolerant) Steane encoder: put the three "pivot" qubits
# of the X-stabilizers (0, 1, 3) into |+>, then fan each one out with CNOTs onto
# the other qubits in its stabilizer.  This realises
# |0_L> ~ prod_g (I + g) |0...0> over the X-stabilizer group.


@guppy
def logical_zero() -> array[qubit, 7]:
    """Allocate seven qubits and encode them into the logical |0> state."""
    q = array(qubit() for _ in range(7))

    # Pivot qubits of the three X-stabilizers -> |+>.
    h(q[0])
    h(q[1])
    h(q[3])

    # g_X3 = X0 X2 X4 X6   (pivot 0)
    cx(q[0], q[2])
    cx(q[0], q[4])
    cx(q[0], q[6])

    # g_X2 = X1 X2 X5 X6   (pivot 1)
    cx(q[1], q[2])
    cx(q[1], q[5])
    cx(q[1], q[6])

    # g_X1 = X3 X4 X5 X6   (pivot 3)
    cx(q[3], q[4])
    cx(q[3], q[5])
    cx(q[3], q[6])

    return q


# ---------------------------------------------------------------------------
# Logical gates (these are the pieces the user wants to read off easily)
# ---------------------------------------------------------------------------


@guppy
def logical_h(q: array[qubit, 7]) -> None:
    """Logical Hadamard: transversal H on all seven data qubits."""
    for i in range(7):
        h(q[i])


@guppy
def logical_s(q: array[qubit, 7]) -> None:
    """Logical phase gate S_L = prod_j Sdg_j (transversal, Sec. II B)."""
    for i in range(7):
        sdg(q[i])


@guppy
def logical_x(q: array[qubit, 7]) -> None:
    """Logical Pauli X ~ X0 X1 X2."""
    x(q[0])
    x(q[1])
    x(q[2])


@guppy
def logical_z(q: array[qubit, 7]) -> None:
    """Logical Pauli Z ~ Z0 Z1 Z2."""
    z(q[0])
    z(q[1])
    z(q[2])


@guppy
def logical_cx(control: array[qubit, 7], target: array[qubit, 7]) -> None:
    """Logical CNOT: transversal physical CNOT(control_j, target_j)."""
    for i in range(7):
        cx(control[i], target[i])


# ---------------------------------------------------------------------------
# Logical Z-basis measurement with software decoding (Sec. II C)
# ---------------------------------------------------------------------------
# Measure all seven data qubits in the Z basis to get a 7-bit string b.  The
# bit-flip syndrome is s = h . b (mod 2); a non-zero syndrome localises a single
# error on qubit int(s) - 1, where int(s) = 4*s0 + 2*s1 + s2.  After correcting
# that bit, b is a valid Hamming codeword, for which the logical outcome equals
# the parity of all seven bits.  Flipping one bit always toggles that parity, so
# the corrected logical bit is simply  (parity of b) XOR (syndrome non-zero).


@guppy
def measure_logical_z(q: array[qubit, 7] @ owned) -> bool:
    """Destructively measure the logical qubit in the Z basis, with decoding."""
    b = measure_array(q)

    # Bit-flip syndrome from the rows of the parity-check matrix h.
    s0 = b[3] ^ b[4] ^ b[5] ^ b[6]
    s1 = b[1] ^ b[2] ^ b[5] ^ b[6]
    s2 = b[0] ^ b[2] ^ b[4] ^ b[6]
    syndrome_nonzero = s0 | s1 | s2

    raw_parity = b[0] ^ b[1] ^ b[2] ^ b[3] ^ b[4] ^ b[5] ^ b[6]
    return raw_parity ^ syndrome_nonzero


# ---------------------------------------------------------------------------
# The example circuit from main.py, run on logical qubits
# ---------------------------------------------------------------------------
# main.py builds a Bell pair, measures q1, and conditionally corrects q2 so that
# q2 deterministically reads 0.  Here the exact same program runs on two Steane
# logical qubits.  Each "logical_*" call below mirrors one line of main.py.


@guppy
def evaluate() -> None:
    q1 = logical_zero()
    q2 = logical_zero()

    logical_h(q1)
    logical_cx(q1, q2)

    outcome = measure_logical_z(q1)
    result("q1", outcome)

    if outcome:
        logical_x(q2)

    result("q2", measure_logical_z(q2))


# ---------------------------------------------------------------------------
# Bare (unencoded) version of the same circuit, for an apples-to-apples
# comparison against the encoded one under the same physical noise.
# ---------------------------------------------------------------------------


@guppy
def evaluate_bare() -> None:
    q1, q2 = qubit(), qubit()

    h(q1)
    cx(q1, q2)

    outcome = measure(q1)
    result("q1", outcome)

    if outcome:
        x(q2)

    result("q2", measure(q2))


evaluate.check()
evaluate_bare.check()


if __name__ == "__main__":
    from steane.faulty_qubits import compare

    compare()
