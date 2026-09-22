"""
Single-shot phase probes for the multicontrolled ops used in qvst.py.

Each probe prepares (|ref> + |fire>)/sqrt(2), applies one multicontrolled op,
and reads the sign/phase of the fire-branch amplitude relative to the
(untouched) reference branch. Selene normalizes so the first nonzero
amplitude is real-positive, which the reference branch provides.

Run from guppy-test root: venv/bin/python -m qubitization.probe_phase
"""

import numpy as np

import guppylang
from guppylang import guppy, qubit
from guppylang.emulator import EmulatorBuilder
from guppylang.std.builtins import array, comptime
from guppylang.std.debug import state_result
from guppylang.std.quantum import cx, discard_array, h, x
from tket.passes import ModifierResolverPass

from qubitization.block_encodings import pauli_term, phase_pi
from qubitization.util import multicontrol

guppylang.enable_experimental_features()


def probe(mc_fn, nc, nt, hmask, xmask, label):
    """
    hmask: bits (big-endian over the nc+nt register) forming a GHZ ladder
    (h on the first set bit, cx to the rest), creating |0..0> + |hmask>.
    xmask: bits X-flipped in both branches.
    Prints nonzero amplitudes of the output state.
    """
    n = nc + nt

    def make():
        @guppy
        def sim() -> None:
            qs = array(qubit() for _ in range(comptime(n)))
            first = -1
            for i in range(comptime(n)):
                if (comptime(hmask) >> (comptime(n) - 1 - i)) & 1:
                    if first < 0:
                        first = i
                        h(qs[i])
                    else:
                        cx(qs[first], qs[i])
            for i in range(comptime(n)):
                if (comptime(xmask) >> (comptime(n) - 1 - i)) & 1:
                    x(qs[i])

            ctrls = array(qs.take(i) for i in range(comptime(nc)))
            tgts = array(qs.take(comptime(nc) + i) for i in range(comptime(nt)))
            mc_fn(ctrls, tgts)
            for i in range(comptime(nc)):
                qs.put(ctrls.take(i), i)
            for i in range(comptime(nt)):
                qs.put(tgts.take(i), comptime(nc) + i)
            ctrls.discard_all_taken()
            tgts.discard_all_taken()

            state_result("out", qs)
            discard_array(qs)

        return sim

    sim = make()
    pack = sim.compile()
    ModifierResolverPass().run(pack.modules[0])
    emu = EmulatorBuilder().build(pack, n_qubits=n)
    st = emu.run().partial_state_dicts()[0]["out"].state_distribution()[0].state
    nz = {format(i, f"0{n}b"): np.round(a, 4) for i, a in enumerate(st) if abs(a) > 1e-8}
    print(f"{label}: {nz}")
    return st


if __name__ == "__main__":
    N = 2  # targets

    # P1: x on t0, nc=3, cs=0b011 (ising k=1 term). Fire: |011,00> -> |011,10>.
    # Input (|000,00> + |011,00>)/sqrt2. Expect amp[01110] = +0.7071; bug -> -0.7071.
    mc = multicontrol(pauli_term(N, [("x", 0)]), N, 3, 0b011)
    probe(mc, 3, N, 0b01100, 0, "P1 x0 cs=011")

    # P1b: same but cs=0b101 (matches test.py's passing 3-control config).
    mc = multicontrol(pauli_term(N, [("x", 0)]), N, 3, 0b101)
    probe(mc, 3, N, 0b10100, 0, "P1b x0 cs=101")

    # P2: zz, nc=3, cs=0b001, targets prepped |01> in both branches.
    # Input (|000,01> + |001,01>)/sqrt2. Fire: ZZ|01> = -|01>.
    # Expect amp[00101] = -0.7071 (amp[00001] = +0.7071 reference).
    mc = multicontrol(pauli_term(N, [("z", 0), ("z", 1)]), N, 3, 0b001)
    probe(mc, 3, N, 0b00100, 0b00001, "P2 zz cs=001 tgt|01>")

    # P3: phase_pi (global -I), nc=3, cs=0b100.
    # Input (|000,00> + |100,00>)/sqrt2. Expect amp[10000] = -0.7071.
    mc = multicontrol(phase_pi(N), N, 3, 0b100)
    probe(mc, 3, N, 0b10000, 0, "P3 phase_pi cs=100")

    # P4: x on t0 with a SINGLE control, cs=1 (plain CX from control to t0).
    # Input (|0,00> + |1,00>)/sqrt2. Expect amp[110] = +0.7071.
    mc = multicontrol(pauli_term(N, [("x", 0)]), N, 1, 1)
    probe(mc, 1, N, 0b100, 0, "P4 x0 nc=1 cs=1")

def probe_y():
    N = 2
    # P5: y on t0, single control, cs=1. Fire: y|00> = i|10>.
    # Input (|0,00> + |1,00>)/sqrt2. Exact -> amp[110] = +0.7071j; CRy(pi)-bug -> +0.7071.
    mc = multicontrol(pauli_term(N, [("y", 0)]), N, 1, 1)
    probe(mc, 1, N, 0b100, 0, "P5 y0 nc=1 cs=1")
