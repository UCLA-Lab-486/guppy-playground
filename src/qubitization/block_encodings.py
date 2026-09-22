"""
This file contains circuits for the block encoded hamiltonians.

Guppy port of the encodings in the Qiskit reference `hamiltonian_qvst.py`:
each Hamiltonian H = sum_k params[k] * P_k (P_k a Pauli string) is described
by an `Encoding` holding the PREPARE circuit on the control register and the
list of Pauli-term circuits used to build SELECT in qvst.py.
"""

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np

import guppylang
from guppylang import guppy, qubit
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import h as h_gate, x, y
from pytket import Circuit
from pytket.circuit import StatePreparationBox
from pytket.passes import DecomposeBoxes

from qubitization.params import J, h
from qubitization.util import vec_to_guppy

guppylang.enable_experimental_features()


# ---------------------------------------------------------------------------
# Pauli-term circuits
# ---------------------------------------------------------------------------
#
# Each factory lives in its own Python function scope so that the comptime
# captures (n, i, ...) are bound at definition time, not late-bound loop
# variables.


def _single_pauli(n: int, gate_name: str, i: int) -> GuppyFunctionDefinition[..., Any]:
    """One Pauli gate (x/y/z) on qubit i of an n-qubit register."""
    if gate_name == "x":

        @guppy(unitary=True)
        def term(qs: array[qubit, comptime(n)]) -> None:
            x(qs[comptime(i)])

    elif gate_name == "y":

        @guppy(unitary=True)
        def term(qs: array[qubit, comptime(n)]) -> None:
            y(qs[comptime(i)])

    elif gate_name == "z":
        # Z = H X H. A bare z under `with control` resolves to CRz(pi), which
        # fires as -i*Z instead of Z (guppylang 0.21.16 / tket 0.13.1
        # ModifierResolverPass); h and x resolve exactly, so build Z from them.

        @guppy(unitary=True)
        def term(qs: array[qubit, comptime(n)]) -> None:
            h_gate(qs[comptime(i)])
            x(qs[comptime(i)])
            h_gate(qs[comptime(i)])

    else:
        raise ValueError(f"unknown gate name {gate_name!r}")

    return term


def _chain_unitary(
    n: int,
    f: GuppyFunctionDefinition[..., Any],
    g: GuppyFunctionDefinition[..., Any],
) -> GuppyFunctionDefinition[..., Any]:
    @guppy(unitary=True)
    def term(qs: array[qubit, comptime(n)]) -> None:
        f(qs)
        g(qs)

    return term


def pauli_term(n: int, ops) -> GuppyFunctionDefinition[..., Any]:
    """
    Circuit for a Pauli string on n qubits.

    `ops` is a sequence of (gate_name, qubit_index) pairs with gate_name in
    {"x", "y", "z"}. Returns a `@guppy(unitary=True)` function on
    `array[qubit, n]` applying those gates in order.
    """
    steps = [_single_pauli(n, gate_name, i) for gate_name, i in ops]
    return functools.reduce(lambda f, g: _chain_unitary(n, f, g), steps)


def phase_pi(n: int) -> GuppyFunctionDefinition[..., Any]:
    """Global phase pi on n qubits: Y X Y X = -I applied to qs[0].

    Built from x and y rather than z because a bare z under `with control`
    fires with an extra -i phase (see _single_pauli); x and y fire exactly.
    """

    @guppy(unitary=True)
    def term(qs: array[qubit, comptime(n)]) -> None:
        x(qs[0])
        y(qs[0])
        x(qs[0])
        y(qs[0])

    return term


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------


@dataclass
class Encoding:
    """Block encoding of H = sum_k params[k] * terms[k] with norm = sum(params)."""

    control_numbers: int
    prep: GuppyFunctionDefinition[..., Any]
    prep_dg: GuppyFunctionDefinition[..., Any]
    terms: list
    norm: float


_prep_counter = 0


def _make_prep_pair(control_numbers: int, state: np.ndarray, label: str):
    """PREPARE circuit and its inverse on the control register."""
    global _prep_counter
    _prep_counter += 1

    prep = vec_to_guppy(
        control_numbers, state, name=f"prep_{label}_{_prep_counter}"
    )

    # util.vec_to_guppy does not expose the dagger of its internal pytket
    # circuit, so replicate the construction here and invert it.
    qc = Circuit(control_numbers)
    box = StatePreparationBox(state)
    qc.add_gate(box, list(range(control_numbers)))
    DecomposeBoxes().apply(qc)
    prep_dg = guppy.load_pytket(f"prep_dg_{label}_{_prep_counter}", qc.dagger())

    return prep, prep_dg


def _prep_state(params: list) -> tuple[int, np.ndarray, float]:
    control_numbers = int(np.ceil(np.log2(len(params))))
    norm = sum(params)
    state = np.zeros(2**control_numbers, dtype=np.complex128)
    for idx in range(len(params)):
        state[idx] = np.sqrt(params[idx] / norm)
    return control_numbers, state, norm


def qvst_ising_chain(num_qubits: int) -> Encoding:
    """Transverse-field Ising chain: J * ZZ pairs + h * X per qubit."""
    params = [J for _ in range(num_qubits - 1)] + [h for _ in range(num_qubits)]
    control_numbers, state, norm = _prep_state(params)
    prep, prep_dg = _make_prep_pair(control_numbers, state, "ising")

    terms = []
    for idx in range(num_qubits - 1):
        terms.append(pauli_term(num_qubits, [("z", idx), ("z", idx + 1)]))
    for idx in range(num_qubits):
        terms.append(pauli_term(num_qubits, [("x", idx)]))

    return Encoding(control_numbers, prep, prep_dg, terms, norm)


def qvst_heis_chain(num_qubits: int) -> Encoding:
    """Heisenberg chain: J * (ZZ + XX + YY) pairs + h * X per qubit."""
    params = [J for _ in range(3 * num_qubits - 3)] + [h for _ in range(num_qubits)]
    control_numbers, state, norm = _prep_state(params)
    prep, prep_dg = _make_prep_pair(control_numbers, state, "heis")

    terms = []
    for idx in range(num_qubits - 1):
        terms.append(pauli_term(num_qubits, [("z", idx), ("z", idx + 1)]))
    for idx in range(num_qubits - 1):
        terms.append(pauli_term(num_qubits, [("x", idx), ("x", idx + 1)]))
    for idx in range(num_qubits - 1):
        terms.append(pauli_term(num_qubits, [("y", idx), ("y", idx + 1)]))
    for idx in range(num_qubits):
        terms.append(pauli_term(num_qubits, [("x", idx)]))

    return Encoding(control_numbers, prep, prep_dg, terms, norm)
