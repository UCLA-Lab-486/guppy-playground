"""
QVST (qubitization) Hamiltonian simulation, guppy port of `benchmark_hs.py`.

Register layout on the full array `qs` (big-endian, ancillas are high bits):
    qs[0]                          sig   (QSP signal qubit; Qiskit's -1)
    qs[1 .. CN]                    creg  (control register, creg[0] = MSB)
    qs[CN + 1]                     anc   (Qiskit's -2)
    qs[CN + 2 .. CN + 1 + N]       sys   (system qubits)

With this layout the all-ancilla-zero block occupies global state indices
0 .. 2^N - 1.
"""

import functools
import math
from typing import Any

import guppylang
from guppylang import guppy, qubit
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import angle, cx, h, rx, s, sdg

from qubitization.block_encodings import phase_pi
from qubitization.util import multicontrol

guppylang.enable_experimental_features()


# ---------------------------------------------------------------------------
# Step factories
#
# Each factory is its own Python function scope so comptime captures are bound
# at definition time. Every step acts on the full `array[qubit, total]`.
# ---------------------------------------------------------------------------


def _chain(total, f, g):
    @guppy
    def step(qs: array[qubit, comptime(total)]) -> None:
        f(qs)
        g(qs)

    return step


def _rx_sig(total, theta_rad):
    # guppy angles are in half-turns: angle(v) rotates by v*pi radians.
    half_turns = theta_rad / math.pi

    @guppy
    def step(qs: array[qubit, comptime(total)]) -> None:
        rx(qs[0], angle(comptime(half_turns)))

    return step


def _s_sig(total):
    @guppy
    def step(qs: array[qubit, comptime(total)]) -> None:
        s(qs[0])

    return step


def _sdg_sig(total):
    @guppy
    def step(qs: array[qubit, comptime(total)]) -> None:
        sdg(qs[0])

    return step


def _h_anc(total, anc_idx):
    @guppy
    def step(qs: array[qubit, comptime(total)]) -> None:
        h(qs[comptime(anc_idx)])

    return step


def _cx_sig_anc(total, anc_idx):
    @guppy
    def step(qs: array[qubit, comptime(total)]) -> None:
        cx(qs[0], qs[comptime(anc_idx)])

    return step


def _lift1(total, fn, offset, m):
    """Apply `fn` (on array[qubit, m]) to qs[offset .. offset + m - 1]."""

    @guppy
    def step(qs: array[qubit, comptime(total)]) -> None:
        sub = array(qs.take(comptime(offset) + i) for i in range(comptime(m)))
        fn(sub)
        for i in range(comptime(m)):
            qs.put(sub.take(i), comptime(offset) + i)
        sub.discard_all_taken()

    return step


def _lift_mc(total, mcfn, c0, nc, t0, nt):
    """Apply `mcfn(controls, targets)` with controls at offset c0, targets at t0."""

    @guppy
    def step(qs: array[qubit, comptime(total)]) -> None:
        ctrls = array(qs.take(comptime(c0) + i) for i in range(comptime(nc)))
        tgts = array(qs.take(comptime(t0) + i) for i in range(comptime(nt)))
        mcfn(ctrls, tgts)
        for i in range(comptime(nc)):
            qs.put(ctrls.take(i), comptime(c0) + i)
        for i in range(comptime(nt)):
            qs.put(tgts.take(i), comptime(t0) + i)
        ctrls.discard_all_taken()
        tgts.discard_all_taken()

    return step


# ---------------------------------------------------------------------------
# QVST algorithm
# ---------------------------------------------------------------------------


def qvst_algo(
    num_qubits: int, time_t: float, encoding_builder
) -> tuple[GuppyFunctionDefinition[..., Any], int, int]:
    """
    Build the QVST circuit for exp(-i H t) on `num_qubits` system qubits.

    Returns (circuit, total, control_numbers) where `circuit` is a guppy
    function on `array[qubit, total]` and total = num_qubits + CN + 2.
    """
    if time_t == 0.0070711:
        Theta_list = [-1.574, 1.5817]
    elif time_t == 0.0070711 * 2:
        Theta_list = [2.8729, -0.3212, -2.9309, 0.2659]
    elif time_t == 0.0070711 * 4:
        Theta_list = [2.6071, -0.5889, -2.6671, 0.5317]
    elif time_t == 0.066948:
        Theta_list = [-0.6741, 2.5806, 0.7772, -2.4675]
    elif time_t == 0.066948 * 2:
        Theta_list = [3.0914, -1.1879, 2.0483, 1.9397, -1.2966, -1.3871, 1.8492, 0.0473]
    elif time_t == 0.066948 * 4:
        Theta_list = [3.0441, -1.2579, 2.0730, 1.9853, -1.3456, -1.4096, 1.9213, 0.0947]
    else:
        raise ValueError("Unkown time_t")

    encoding = encoding_builder(num_qubits)
    CN = encoding.control_numbers
    N = num_qubits
    total = N + CN + 2
    anc_idx = CN + 1
    sys_off = CN + 2

    # mcz fires iff sig = 1 and creg = |0...0>: controls are qs[0 .. CN]
    # (sig then creg), control_state = 2^CN big-endian.
    mcz = multicontrol(phase_pi(N), n_targets=N, n_controls=CN + 1, control_state=2**CN)
    mcz_step = _lift_mc(total, mcz, 0, CN + 1, sys_off, N)

    # SELECT: term k fires iff anc = 1 and creg = |k>: controls are
    # qs[1 .. CN + 1] (creg then anc), control_state = 2k + 1 big-endian.
    select_steps = []
    for k, term in enumerate(encoding.terms):
        mck = multicontrol(term, n_targets=N, n_controls=CN + 1, control_state=2 * k + 1)
        select_steps.append(_lift_mc(total, mck, 1, CN + 1, sys_off, N))

    prep_step = _lift1(total, encoding.prep, 1, CN)
    prep_dg_step = _lift1(total, encoding.prep_dg, 1, CN)
    s_step = _s_sig(total)
    sdg_step = _sdg_sig(total)
    h_step = _h_anc(total, anc_idx)
    cx_step = _cx_sig_anc(total, anc_idx)

    steps = []
    for idx in range(len(Theta_list)):
        if idx % 2 == 0:
            if idx == 0:
                steps.append(_rx_sig(total, -Theta_list[idx]))
            else:
                steps.append(
                    _rx_sig(total, -Theta_list[idx] + Theta_list[idx - 1] + math.pi)
                )
            steps.append(mcz_step)
            steps.append(s_step)
            steps.append(h_step)
            steps.append(prep_step)
            steps.extend(select_steps)
            steps.append(cx_step)
        else:
            steps.append(_rx_sig(total, Theta_list[idx - 1] - Theta_list[idx] - math.pi))
            steps.append(sdg_step)
            steps.append(cx_step)
            steps.extend(select_steps)
            steps.append(h_step)
            steps.append(prep_dg_step)
            steps.append(mcz_step)

    steps.append(_rx_sig(total, Theta_list[-1] + math.pi))

    circuit = functools.reduce(lambda f, g: _chain(total, f, g), steps)
    return circuit, total, CN
