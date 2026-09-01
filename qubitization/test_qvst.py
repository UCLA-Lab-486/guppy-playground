"""
Test harness for the guppylang qubitization (QVST Hamiltonian simulation) port.

Run with:
    venv/bin/python -m qubitization.test_qvst

Conventions (matching selene's big-endian state output, qs[0] = most
significant bit): the qvst register layout is
    qs[0]                 signal ancilla
    qs[1 .. CN]           control register
    qs[CN + 1]            ancilla
    qs[CN + 2 .. total-1] system (N qubits, sys[0] = MSB of the block index)
so with all ancillas in |0> the system basis states occupy global state
indices 0 .. 2^N - 1, indexed by the system value read big-endian.
"""

import numpy as np
from scipy.linalg import expm

from guppylang import guppy, comptime
from guppylang.emulator import EmulatorBuilder
from guppylang.std.builtins import array
from guppylang.std.debug import state_result
from guppylang.std.quantum import cx, discard_array, h, qubit, t, x
from tket.passes import ModifierResolverPass

from qubitization.util import Mat
from qubitization.params import num_qubits, time_t, J, h as h_field
from qubitization.qvst import qvst_algo
from qubitization.block_encodings import qvst_ising_chain, qvst_heis_chain


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------


def extract_block(circ, total, n_sys):
    """
    Extract the (2^n_sys x 2^n_sys) block of `circ`'s unitary restricted to
    the "all ancillas |0>" rows and columns, with correct relative phases
    between columns (up to one overall global phase).

    `circ` is a @guppy function taking `array[qubit, total]`. The system
    register is the last n_sys qubits, big-endian: bit (n_sys - 1 - i) of the
    block index corresponds to qs[total - n_sys + i].

    Selene reports each output state only up to a global phase, so running on
    basis states alone loses the relative phases between columns. Column b's
    phase relative to column 0 is recovered by additionally running on the
    input (|0> + |b>)/sqrt(2): the output is (c_0 + c_b)/sqrt(2) up to a
    phase e^{i psi}, and since full unitary columns are orthonormal,
    <c_0, c_0 + c_b> = 1, which pins down e^{i psi}.
    """

    def run_state(basis_idx, superpose):
        # Fresh closure scope per (basis_idx, superpose) so the comptime
        # captures below are not late-bound.
        @guppy
        def sim() -> None:
            qs = array(qubit() for _ in range(comptime(total)))

            # Prepare |ancillas=0, sys=basis_idx>, or the superposition
            # (|0> + |basis_idx>)/sqrt(2) via a GHZ-style ladder (H on the
            # first set bit, CX to the rest) when superpose is set.
            first = -1
            for i in range(comptime(n_sys)):
                if (comptime(basis_idx) >> (comptime(n_sys) - 1 - i)) & 1:
                    q = comptime(total) - comptime(n_sys) + i
                    if comptime(superpose):
                        if first < 0:
                            first = q
                            h(qs[q])
                        else:
                            cx(qs[first], qs[q])
                    else:
                        x(qs[q])

            circ(qs)

            state_result("out_state", qs)
            discard_array(qs)

        # `with control(...)` blocks compile to modifier ops that selene
        # cannot execute directly; resolve them first (harmless otherwise).
        pack = sim.compile()
        ModifierResolverPass().run(pack.modules[0])
        emu = EmulatorBuilder().build(pack, n_qubits=total)

        dist = emu.run().partial_state_dicts()[0]["out_state"].state_distribution()
        return dist[0].state

    dim = 2**n_sys
    block = np.zeros((dim, dim), dtype=np.complex128)

    c0 = run_state(0, False)
    block[:, 0] = c0[:dim]

    for b in range(1, dim):
        w = run_state(b, True)
        overlap = np.vdot(c0, w)
        assert abs(overlap) > 1e-6, (
            f"column {b}: overlap with column 0 too small ({abs(overlap):.2e}); "
            "cannot pin the relative phase"
        )
        phase = 1.0 / (np.sqrt(2) * overlap)
        column_full = np.sqrt(2) * phase * w - c0
        block[:, b] = column_full[:dim]

    return block


def assert_allclose_up_to_global_phase(got, want, msg=""):
    tr = np.trace(want.conj().T @ got)
    phase = tr / abs(tr) if abs(tr) > 1e-9 else 1.0
    assert np.allclose(got, phase * want), f"{msg}\n{got}\n!=\n{want}"


# ---------------------------------------------------------------------------
# Extraction self-test (independent of qvst.py / block_encodings.py)
# ---------------------------------------------------------------------------


@guppy
def _selftest_circ(qs: array[qubit, 3]) -> None:
    # Acts only on the "system" qs[1], qs[2]; leaves the ancilla qs[0] alone.
    h(qs[1])
    cx(qs[1], qs[2])
    t(qs[2])


def test_extract_block():
    got = extract_block(_selftest_circ, 3, 2)

    # Big-endian: sys[0] = qs[1] is the leftmost kron factor.
    cx_mat = np.eye(4, dtype=np.complex128)
    cx_mat[2:, 2:] = Mat.X
    want = np.kron(np.eye(2), Mat.T) @ cx_mat @ np.kron(Mat.H, np.eye(2))

    assert_allclose_up_to_global_phase(got, want, "extract_block self-test:")


# ---------------------------------------------------------------------------
# QVST Hamiltonian simulation tests
# ---------------------------------------------------------------------------


def _term(n, ops_at):
    """Kron product over n sites (big-endian, site 0 leftmost) with the 2x2
    operators in `ops_at` (a dict site -> matrix) and identity elsewhere."""
    out = np.array([[1.0 + 0.0j]])
    for i in range(n):
        out = np.kron(out, ops_at.get(i, Mat.I))
    return out


def ising_hamiltonian(n):
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    for i in range(n - 1):
        H += J * _term(n, {i: Mat.Z, i + 1: Mat.Z})
    for i in range(n):
        H += h_field * _term(n, {i: Mat.X})
    return H, J * (n - 1) + h_field * n


def heis_hamiltonian(n):
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    for P in (Mat.Z, Mat.X, Mat.Y):
        for i in range(n - 1):
            H += J * _term(n, {i: P, i + 1: P})
    for i in range(n):
        H += h_field * _term(n, {i: Mat.X})
    return H, 3 * J * (n - 1) + h_field * n


def run_qvst_case(name, encoding_builder, H_mat, norm):
    circuit, total, control_numbers = qvst_algo(num_qubits, time_t, encoding_builder)
    print(
        f"{name}: num_qubits={num_qubits}, time_t={time_t}, "
        f"control_numbers={control_numbers}, total={total}, norm={norm}"
    )

    got = extract_block(circuit, total, num_qubits)
    target = expm(-1j * H_mat / norm * time_t)

    diag = np.diag(got @ target.conj().T)
    print(f"{name} diag(got @ target^dag):", diag)
    print(f"{name} abs(diag):            ", np.abs(diag))

    assert np.all(np.abs(diag) > 0.999), f"{name}: abs(diag) too small: {np.abs(diag)}"

    tr = np.trace(target.conj().T @ got)
    assert abs(tr) > 1e-9, f"{name}: got is orthogonal to target"
    phase = tr / abs(tr)
    err = np.max(np.abs(got - phase * target))
    print(f"{name} max elementwise error (phase-aligned): {err:.3e}")
    assert err < 1e-3, f"{name}: max elementwise error {err:.3e} >= 1e-3"


def test_qvst_ising():
    H_mat, norm = ising_hamiltonian(num_qubits)
    run_qvst_case("ising", qvst_ising_chain, H_mat, norm)


def test_qvst_heis():
    H_mat, norm = heis_hamiltonian(num_qubits)
    run_qvst_case("heis", qvst_heis_chain, H_mat, norm)


if __name__ == "__main__":
    test_extract_block()
    print("test_extract_block passed")
    test_qvst_ising()
    print("test_qvst_ising passed")
    test_qvst_heis()
    print("test_qvst_heis passed")
    print("all qvst tests passed")
