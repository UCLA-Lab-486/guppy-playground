import numpy as np

from guppylang import guppy, comptime
from guppylang.emulator import EmulatorBuilder
from guppylang.std.builtins import result, array
from guppylang.std.debug import state_result
from guppylang.std.quantum import cx, discard_array, h, measure, qubit, t, x
from guppylang.std.num import nat
from tket.passes import ModifierResolverPass

from qubitization.util import Mat, guppy_to_unitary, multicontrol, vec_to_guppy


@guppy
def simple_circuit(qs: array[qubit, 2]) -> None:
    h(qs[0])
    cx(qs[0], qs[1])


params = [1] * 5
a = comptime(int(np.ceil(np.log2(len(params)))))
print(a)

print(guppy_to_unitary(2, simple_circuit))

initial_state = np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0], dtype=np.complex128)

qc = vec_to_guppy(2, initial_state, name="my_init")
print(qc)
unitary = guppy_to_unitary(2, qc)
print(unitary)


# ---------------------------------------------------------------------------
# multicontrol tests
#
# Conventions (matching selene's state output, where qs[0] is the most
# significant bit): the combined register is |controls, targets>, so basis
# index = c * 2^n_targets + t with c and t read big-endian over their
# registers. The expected unitary is therefore block diagonal, with block c
# equal to U when c == control_state and identity otherwise.
# ---------------------------------------------------------------------------


def multicontrol_unitary(mc_fn, n_controls, n_targets):
    """
    Extract the unitary of a multicontrol guppy function
    fn(controls: array[qubit, n_controls], targets: array[qubit, n_targets])
    on the combined register |controls, targets> (big-endian).

    Selene reports each output state only up to a global phase, so applying
    mc_fn to basis states alone loses the relative phases between columns.
    We recover column b's phase relative to column 0 by additionally running
    on the input (|0> + |b>)/sqrt(2): the output is (c_0 + c_b)/sqrt(2) up to
    a phase e^{i psi}, and since unitary columns are orthonormal,
    <c_0, c_0 + c_b> = 1, which pins down e^{i psi}. The result is the true
    unitary up to one overall global phase (that of column 0).
    """
    n = n_controls + n_targets

    def run_state(basis_idx, superpose):
        @guppy
        def sim() -> None:
            qs = array(qubit() for _ in range(comptime(n)))

            # Prepare |basis_idx>, or (|0> + |basis_idx>)/sqrt(2) via a
            # GHZ-style ladder over the set bits when superpose is set.
            first = -1
            for i in range(comptime(n)):
                if (comptime(basis_idx) >> (comptime(n) - 1 - i)) & 1:
                    if comptime(superpose):
                        if first < 0:
                            first = i
                            h(qs[i])
                        else:
                            cx(qs[first], qs[i])
                    else:
                        x(qs[i])

            ctrls = array(qs.take(i) for i in range(comptime(n_controls)))
            tgts = array(
                qs.take(comptime(n_controls) + i) for i in range(comptime(n_targets))
            )

            mc_fn(ctrls, tgts)

            for i in range(comptime(n_controls)):
                qs.put(ctrls.take(i), i)
            for i in range(comptime(n_targets)):
                qs.put(tgts.take(i), comptime(n_controls) + i)
            ctrls.discard_all_taken()
            tgts.discard_all_taken()

            state_result("out_state", qs)
            discard_array(qs)

        # The `with control(...)` block inside multicontrol compiles to a
        # modifier op that selene cannot execute directly; resolve it first.
        pack = sim.compile()
        ModifierResolverPass().run(pack.modules[0])
        emu = EmulatorBuilder().build(pack, n_qubits=n)

        dist = emu.run().partial_state_dicts()[0]["out_state"].state_distribution()
        return dist[0].state

    unitary_matrix = np.zeros((2**n, 2**n), dtype=np.complex128)
    c0 = run_state(0, False)
    unitary_matrix[:, 0] = c0

    for basis_idx in range(1, 2**n):
        w = run_state(basis_idx, True)
        phase = 1.0 / (np.sqrt(2) * np.vdot(c0, w))
        unitary_matrix[:, basis_idx] = np.sqrt(2) * phase * w - c0

    return unitary_matrix


def assert_allclose_up_to_global_phase(got, want, msg=""):
    tr = np.trace(want.conj().T @ got)
    phase = tr / abs(tr) if abs(tr) > 1e-9 else 1.0
    assert np.allclose(got, phase * want), f"{msg}\n{got}\n!=\n{want}"


def expected_multicontrol(U_mat, n_controls, control_state):
    """Block-diagonal expected unitary: U on the |control_state> block, I elsewhere."""
    dim = U_mat.shape[0]
    full = np.zeros((2**n_controls * dim, 2**n_controls * dim), dtype=np.complex128)
    for c in range(2**n_controls):
        block = U_mat if c == control_state else np.eye(dim)
        full[c * dim : (c + 1) * dim, c * dim : (c + 1) * dim] = block
    return full


@guppy(unitary=True)
def u_x(qs: array[qubit, 1]) -> None:
    x(qs[0])


@guppy(unitary=True)
def u_h(qs: array[qubit, 1]) -> None:
    h(qs[0])


@guppy(unitary=True)
def u_bell_t(qs: array[qubit, 2]) -> None:
    h(qs[0])
    cx(qs[0], qs[1])
    t(qs[1])


# Big-endian matrices for the target circuits above.
u_x_mat = Mat.X
u_h_mat = Mat.H
cx_mat = np.eye(4, dtype=np.complex128)
cx_mat[2:, 2:] = Mat.X
u_bell_t_mat = np.kron(np.eye(2), Mat.T) @ cx_mat @ np.kron(Mat.H, np.eye(2))


def test_single_control_all_states():
    # 1 control, 1 target: control_state=1 is a plain CX, control_state=0 an
    # anti-controlled X.
    for control_state in range(2):
        mc = multicontrol(u_x, 1, 1, control_state)
        got = multicontrol_unitary(mc, 1, 1)
        want = expected_multicontrol(u_x_mat, 1, control_state)
        assert_allclose_up_to_global_phase(
            got, want, f"1-control X, control_state={control_state}:"
        )


def test_two_controls_all_states():
    # 2 controls, 1 target, all 4 control states. The asymmetric states 0b01
    # and 0b10 verify the bit-ordering convention.
    for control_state in range(4):
        mc = multicontrol(u_h, 1, 2, control_state)
        got = multicontrol_unitary(mc, 2, 1)
        want = expected_multicontrol(u_h_mat, 2, control_state)
        assert_allclose_up_to_global_phase(
            got, want, f"2-control H, control_state={control_state}:"
        )


def test_three_controls():
    # 3 controls, 1 target: shows the implementation is parameterizable in the
    # number of control qubits via comptime.
    control_state = 0b101
    mc = multicontrol(u_x, 1, 3, control_state)
    got = multicontrol_unitary(mc, 3, 1)
    want = expected_multicontrol(u_x_mat, 3, control_state)
    assert_allclose_up_to_global_phase(got, want, "3-control X:")


def test_multi_qubit_target():
    # 2 controls, 2-qubit entangling target circuit.
    control_state = 0b10
    mc = multicontrol(u_bell_t, 2, 2, control_state)
    got = multicontrol_unitary(mc, 2, 2)
    want = expected_multicontrol(u_bell_t_mat, 2, control_state)
    assert_allclose_up_to_global_phase(got, want, "2-control bell+T:")


if __name__ == "__main__":
    test_single_control_all_states()
    print("test_single_control_all_states passed")
    test_two_controls_all_states()
    print("test_two_controls_all_states passed")
    test_three_controls()
    print("test_three_controls passed")
    test_multi_qubit_target()
    print("test_multi_qubit_target passed")
    print("all multicontrol tests passed")
