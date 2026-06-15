from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, h, measure, qubit, x


@guppy
def simple_circuit() -> qubit:
    q1, q2 = qubit(), qubit()

    h(q1)
    cx(q1, q2)

    outcome = measure(q1)
    result("q1", outcome)

    if outcome:
        x(q2)

    return q2

simple_circuit.check()

@guppy
def evaluate() -> None:
    q = simple_circuit()
    result("q2", measure(q))

emulator = evaluate.emulator(n_qubits=2).stabilizer_sim().with_seed(3)
sim_result = emulator.run()
print(list(sim_result.results))

import matplotlib.pyplot as plt
import numpy as np

shots = evaluate.emulator(n_qubits=2).with_seed(0).with_shots(10000).run()

fig, ax = plt.subplots(1, 1)
possible_outcomes = ["00", "01", "10", "11"]
idx = np.asarray(list(range(len(possible_outcomes))))
counts = [len([1 for shot in shots if str(shot.as_dict()['q1']) + str(shot.as_dict()['q2']) == o]) for o in possible_outcomes]

bars = ax.bar(idx, counts)
ax.bar_label(bars, labels=counts)

ax.set_title("Circuit simulation")
ax.set_xlabel("Measurement outcomes")
ax.set_xticks(idx)
ax.set_xticklabels(possible_outcomes)
ax.set_ylabel("Frequency")
plt.show()
