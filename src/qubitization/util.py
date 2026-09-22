import guppylang
from guppylang import guppy, qubit
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime, control
from guppylang.std.debug import state_result
from guppylang.std.quantum import cx, h, measure, qubit, x, discard_array
import numpy as np
from numpy.typing import NDArray
from pytket import Circuit
from pytket.circuit import StatePreparationBox
from pytket.passes import DecomposeBoxes
from typing import Any
import math

# Common operators

class Mat:
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = 1 / math.sqrt(2) * np.array([[1, 1], [1, -1]])
    Sdg = (np.array([[1, 0],
                     [0, 0]]) + 1j *
           np.array([[0, 0],
                     [0, -1]]))
    S = (np.array([[1, 0],
                   [0, 0]]) + 1j *
         np.array([[0, 0],
                   [0, 1]]))
    Tdg = np.array([[1, 0],
                    [0, np.exp(-1j * np.pi / 4)]])
    T = np.array([[1, 0],
                  [0, np.exp(1j * np.pi / 4)]])


def guppy_to_unitary(n: int, circuit: GuppyFunctionDefinition) -> NDArray[np.complex128]:
    """
    Extracts the unitary matrix of an n-qubit Guppy circuit.

    Args:
        circuit_func: A @guppy function that takes an `array[qubit, n]` 
                      and returns None (borrows the qubits).
        n: Number of qubits.

    Returns:
        A (2^n, 2^n) complex numpy array representing the unitary.
    """
    unitary_matrix = np.zeros((2**n, 2**n), dtype=np.complex128)

    for basis_idx in range(2**n):

        @guppy
        def sim_basis_state() -> None:
            qs: array[qubit, comptime(n)] = array(qubit() for _ in range(comptime(n)))

            for i in range(comptime(n)):
                if (comptime(basis_idx) >> i) & 1:
                    x(qs[i])

            circuit(qs)

            state_result("out_state", qs)
            discard_array(qs)

        sim = sim_basis_state.emulator(n_qubits=n)
        result = sim.run()

        # Get the state result dictionaries for the first (and only) shot
        states_dict = result.partial_state_dicts()[0]

        # Get the probabilistic state distribution for our "out_state" tag
        dist = states_dict["out_state"].state_distribution()

        # Since this is a pure unitary without measurements, there is exactly
        # one state in the distribution with probability 1.0. We extract its
        # underlying numpy array.
        state_vec = dist[0].state

        unitary_matrix[:, basis_idx] = state_vec

    return unitary_matrix


def vec_to_guppy(n: int, state_vec: np.ndarray, *, name: str = "prep_fn") -> GuppyFunctionDefinition[..., Any]:
    """
    Given an initial state vector of n qubits, return a guppy circuit that
    prepares that initial state.
    """

    if state_vec.shape != (2**n,):
        raise ValueError(f"state_vec must be a vector of size 2^{n}, got {state_vec.shape}")

    qc = Circuit(n)
    box = StatePreparationBox(state_vec)
    qc.add_gate(box, list(range(n)))
    DecomposeBoxes().apply(qc)

    guppy_circuit = guppy.load_pytket(name, qc)

    return guppy_circuit


def multicontrol(
    U: GuppyFunctionDefinition[..., Any],
    n_targets: int,
    n_controls: int,
    control_state: int,
) -> GuppyFunctionDefinition[..., Any]:
    """
    Guppy analogue of `multicontrol_qiskit` from utils_func.py.

    Given a guppy circuit `U` on `n_targets` qubits, returns a guppy function

        fn(controls: array[qubit, n_controls], targets: array[qubit, n_targets])

    that applies `U` to `targets` iff the control register is in the
    computational basis state `control_state`.

    `U` must be declared with `@guppy(unitary=True)` (or `control=True`) so
    that it is allowed inside a `with control(...)` block.

    `control_state` is read big-endian over the control register: bit
    (n_controls - 1 - i) of `control_state` corresponds to controls[i], so it
    matches the ket |controls[0] ... controls[n_controls - 1]>.

    Note: the returned function uses the (experimental) `with control(...)`
    modifier, which must be lowered with `tket.passes.ModifierResolverPass`
    before the compiled HUGR can be emulated with selene.
    """
    if not 0 <= control_state < 2**n_controls:
        raise ValueError(
            f"control_state must be in [0, 2^{n_controls}), got {control_state}"
        )

    guppylang.enable_experimental_features()

    @guppy
    def multicontrolled(
        controls: array[qubit, comptime(n_controls)],
        targets: array[qubit, comptime(n_targets)],
    ) -> None:
        # Flip the controls whose control_state bit is 0, so that the block
        # below fires exactly when the register is in |control_state>.
        for i in range(comptime(n_controls)):
            if ((comptime(control_state) >> (comptime(n_controls) - 1 - i)) & 1) == 0:
                x(controls[i])

        with control(controls):
            U(targets)

        for i in range(comptime(n_controls)):
            if ((comptime(control_state) >> (comptime(n_controls) - 1 - i)) & 1) == 0:
                x(controls[i])

    return multicontrolled