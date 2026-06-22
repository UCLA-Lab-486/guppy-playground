"""Deterministic, noiseless checks that the Steane decoder corrects errors.

These run without any error model: errors are injected by hand with explicit
gates, so the outcomes are deterministic and the decoder's behaviour is exact.
"""

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import x

from steane.steane import logical_zero, measure_logical_z


@guppy
def inject_single() -> None:
    """|0_L> with a single bit-flip on every data qubit, one shot per qubit.

    A weight-1 X error must always be corrected, so each decoded value is 0.
    """
    for k in range(7):
        q = logical_zero()
        x(q[k])  # single physical bit-flip
        result("corrected_single", measure_logical_z(q))


@guppy
def inject_double() -> None:
    """|0_L> with a weight-2 X error (qubits 0 and 1).

    The code distance is 3, so it corrects 1 error but not 2: this miscorrects
    to the logical-X partner and decodes to 1, confirming the decoder is real.
    """
    q = logical_zero()
    x(q[0])
    x(q[1])
    result("miscorrected_double", measure_logical_z(q))


if __name__ == "__main__":
    inject_single.check()
    inject_double.check()

    single = inject_single.emulator(n_qubits=7).with_seed(0).run()
    vals = [v for _, v in single.results[0].entries]
    assert all(v == 0 for v in vals), f"single errors not corrected: {vals}"
    print(f"single bit-flip on each of 7 qubits -> all corrected to 0: {vals}")

    double = inject_double.emulator(n_qubits=7).with_seed(0).run()
    dval = double.results[0].entries[0][1]
    assert dval == 1, f"weight-2 error should miscorrect to 1, got {dval}"
    print(f"weight-2 error (qubits 0,1) -> miscorrected to {dval} (expected 1)")

    print("\nOK: decoder corrects all single errors and is fooled by weight-2.")
