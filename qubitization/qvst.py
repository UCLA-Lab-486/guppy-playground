import numpy as np

from guppylang import guppy, comptime
from guppylang.std.builtins import result, array
from guppylang.std.quantum import cx, h, measure, qubit, x, z
from params import J
from qubitization.util import vec_to_guppy

@guppy
def 

def qvst_heis_chain(num_qubits: int):
    params = [J for _ in range(3 * num_qubits - 3)] + [h for _ in range(num_qubits)]
    control_numbers = comptime(int(np.ceil(np.log2(len(params)))))

    norm = sum(params)

    state = np.zeros((2 ** control_numbers)) + 1j * np.zeros((2 ** control_numbers))

    for idx in range(len(params)):
        state[idx] = np.sqrt(params[idx] / norm)

    qc_prep = vec_to_guppy(control_numbers, state)

    @guppy
    def qc_U(qs: array[qubit, comptime(num_qubits)]) -> None:
        """
        Implement qc_U from below in Guppy
        """

        # ZXZX(qs[0]) implements a global phase of pi
        x(qs[0])
        z(qs[0])
        x(qs[0])
        z(qs[0])


        # Claude: use multicontrol here

        
    qc_U = QuantumCircuit(num_qubits)
    qc_U.global_phase = np.pi
    mczgate = multicontrol_qiskit(qc_U, control_numbers + 1, 1)

    control_hamiltonian = QuantumCircuit(num_qubits + control_numbers + 1)
    cnt = 0

    for idx in range(num_qubits - 1):
        qc_U = QuantumCircuit(num_qubits)
        qc_U.z(idx)
        qc_U.z(idx + 1)
        control_hamiltonian.compose(multicontrol_qiskit(qc_U, control_numbers + 1, cnt + 2 ** control_numbers),
                   [idx for idx in range(num_qubits + control_numbers + 1)], inplace=True)
        cnt += 1

    for idx in range(num_qubits - 1):
        qc_U = QuantumCircuit(num_qubits)
        qc_U.x(idx)
        qc_U.x(idx + 1)
        control_hamiltonian.compose(multicontrol_qiskit(qc_U, control_numbers + 1, cnt + 2 ** control_numbers),
                   [idx for idx in range(num_qubits + control_numbers + 1)], inplace=True)
        cnt += 1

    for idx in range(num_qubits - 1):
        qc_U = QuantumCircuit(num_qubits)
        qc_U.y(idx)
        qc_U.y(idx + 1)
        control_hamiltonian.compose(multicontrol_qiskit(qc_U, control_numbers + 1, cnt + 2 ** control_numbers),
                   [idx for idx in range(num_qubits + control_numbers + 1)], inplace=True)
        cnt += 1

    for idx in range(num_qubits):
        qc_U = QuantumCircuit(num_qubits)
        qc_U.x(idx)
        control_hamiltonian.compose(multicontrol_qiskit(qc_U, control_numbers + 1, cnt + 2 ** control_numbers),
                   [idx for idx in range(num_qubits + control_numbers + 1)], inplace=True)
        cnt += 1

    return control_numbers, preperation, mczgate, control_hamiltonian
