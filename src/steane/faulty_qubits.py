"""Faulty-qubit testing framework for the Steane code.

Guppy already ships the tooling we need: its Selene emulator accepts pluggable
*error models*.  The bundled ``DepolarizingErrorModel`` injects depolarizing
noise after every operation, with independent failure probabilities for

    * p_1q   - single-qubit gates,
    * p_2q   - two-qubit gates,
    * p_meas - measurements,
    * p_init - qubit initialisation.

This module wraps that model behind a single ``failure_rate`` knob (all four
channels scaled together) so a circuit can be run on emulated faulty physical
qubits, and provides a sweep that compares the Steane-encoded circuit against
the bare, unencoded circuit under identical noise.
"""

from __future__ import annotations

from collections import Counter

from selene_sim.backends.bundled_error_models import DepolarizingErrorModel

from steane.steane import evaluate, evaluate_bare


def run_faulty(circuit, n_qubits, failure_rate, shots=2000, seed=0, scale=None):
    """Run ``circuit`` on emulated faulty physical qubits.

    ``failure_rate`` sets the depolarizing probability for every channel.  Pass
    ``scale`` to weight the channels individually, e.g.
    ``scale={"p_2q": 1.0, "p_1q": 0.1, "p_meas": 1.0, "p_init": 0.1}``; each
    entry multiplies ``failure_rate`` for that channel (default: all 1.0).

    Returns the list of per-shot result dictionaries.
    """
    scale = scale or {}
    error_model = DepolarizingErrorModel(
        p_1q=failure_rate * scale.get("p_1q", 1.0),
        p_2q=failure_rate * scale.get("p_2q", 1.0),
        p_meas=failure_rate * scale.get("p_meas", 1.0),
        p_init=failure_rate * scale.get("p_init", 1.0),
        random_seed=seed,
    )
    shots_result = (
        circuit.emulator(n_qubits=n_qubits)
        .with_error_model(error_model)
        .with_seed(seed)
        .with_shots(shots)
        .run()
    )
    return [shot.as_dict() for shot in shots_result]


def logical_error_rate(shots):
    """Fraction of shots whose final logical readout ``q2`` is wrong.

    In the noiseless circuit ``q2`` is deterministically 0 (see ``main.py``), so
    any non-zero reading is a logical error.
    """
    n_wrong = sum(1 for s in shots if s["q2"] != 0)
    return n_wrong / len(shots)


def outcome_histogram(shots):
    """Counts of the ``q1 q2`` outcome string, matching main.py's histogram."""
    return Counter(f"{s['q1']}{s['q2']}" for s in shots)


def sweep(failure_rates, shots, seed, scale=None):
    """Return [(p, bare_err, steane_err), ...] over the given failure rates."""
    rows = []
    for p in failure_rates:
        bare = run_faulty(evaluate_bare, 2, p, shots=shots, seed=seed, scale=scale)
        steane = run_faulty(evaluate, 2 * 7, p, shots=shots, seed=seed, scale=scale)
        rows.append((p, logical_error_rate(bare), logical_error_rate(steane)))
    return rows


def _print_table(title, rows):
    print(title)
    print(f"{'failure rate':>13} | {'bare P(q2!=0)':>14} | {'steane P(q2!=0)':>16}")
    print("-" * 51)
    for p, bare_err, steane_err in rows:
        winner = "steane" if steane_err < bare_err else "bare"
        print(f"{p:>13.4f} | {bare_err:>14.4f} | {steane_err:>16.4f}   <- {winner}")
    print()


def compare(
    failure_rates=(0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05),
    shots=4000,
    seed=0,
):
    """Run two experiments comparing the Steane-encoded circuit to the bare one.

    Experiment A applies the full depolarizing model to every channel; this is
    the realistic, end-to-end cost of a *non-fault-tolerant* encode-and-decode.
    Experiment B keeps only readout (measurement) noise, which isolates the
    [[7, 1, 3]] decoder and shows it correcting single errors (~p^2 scaling).
    """
    # --- Experiment A: full physical noise on every channel ---------------
    rows_full = sweep(failure_rates, shots, seed)
    _print_table(
        f"[A] Full depolarizing noise, all channels ({shots} shots/point)",
        rows_full,
    )
    print(
        "    The encoded circuit uses 14 data qubits and ~25 two-qubit gates vs.\n"
        "    one CNOT in the bare circuit, and decodes only once at the end. With\n"
        "    a non-fault-tolerant encoder, mid-circuit faults outpace that single\n"
        "    correction -- exactly the gap the paper closes with FT gadgets.\n"
    )

    # --- Experiment B: readout noise only, isolating the decoder ----------
    readout_only = {"p_1q": 0.0, "p_2q": 0.0, "p_init": 0.0, "p_meas": 1.0}
    readout_rates = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15)
    rows_meas = sweep(readout_rates, shots, seed, scale=readout_only)
    _print_table(
        f"[B] Readout noise only -- isolates the [[7,1,3]] decoder ({shots} shots/point)",
        rows_meas,
    )
    print(
        "    Here the encoder and logical gates are noiseless, so only single-bit\n"
        "    readout flips occur. The decoder fixes any single flip, so the logical\n"
        "    error rate falls as ~p^2 and beats the bare qubit below a pseudo-\n"
        "    threshold -- a direct demonstration of error correction.\n"
    )

    # Detail at one representative noise level, in main.py's histogram style.
    p_demo = 0.01
    print(f"Outcome histograms at full noise {p_demo} (ideal: q2 always 0):")
    bare = run_faulty(evaluate_bare, 2, p_demo, shots=shots, seed=seed)
    steane = run_faulty(evaluate, 2 * 7, p_demo, shots=shots, seed=seed)
    print("  bare  :", dict(sorted(outcome_histogram(bare).items())))
    print("  steane:", dict(sorted(outcome_histogram(steane).items())))

    try:
        _plot(rows_full, rows_meas, bare, steane, p_demo)
    except Exception as exc:  # plotting is optional / may be headless
        print(f"\n(skipped plot: {exc})")

    return rows_full, rows_meas


def _plot(rows_full, rows_meas, bare_shots, steane_shots, p_demo):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4))

    for ax, rows, title in (
        (ax0, rows_full, "[A] full depolarizing noise"),
        (ax1, rows_meas, "[B] readout noise only"),
    ):
        ps = [r[0] for r in rows]
        ax.plot(ps, [r[1] for r in rows], "o-", label="bare (1 qubit)")
        ax.plot(ps, [r[2] for r in rows], "s-", label="Steane (7 qubits)")
        ax.plot(ps, ps, "k--", alpha=0.4, label="y = x (break-even)")
        ax.set_xlabel("physical failure rate")
        ax.set_ylabel("logical error rate  P(q2 != 0)")
        ax.set_title(title)
        ax.legend()

    outcomes = ["00", "01", "10", "11"]
    idx = range(len(outcomes))
    bh = outcome_histogram(bare_shots)
    sh = outcome_histogram(steane_shots)
    width = 0.4
    ax2.bar([i - width / 2 for i in idx], [bh.get(o, 0) for o in outcomes],
            width, label="bare")
    ax2.bar([i + width / 2 for i in idx], [sh.get(o, 0) for o in outcomes],
            width, label="steane")
    ax2.set_xticks(list(idx))
    ax2.set_xticklabels(outcomes)
    ax2.set_xlabel("outcome  q1 q2")
    ax2.set_ylabel("frequency")
    ax2.set_title(f"outcomes at full noise {p_demo}")
    ax2.legend()

    fig.tight_layout()
    out = "comparison.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    compare()
