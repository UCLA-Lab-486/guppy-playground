from qiskit import QuantumCircuit, quantum_info
from qiskit.transpiler.passes.synthesis import HLSConfig
from qiskit.circuit.library import StatePreparation, SGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation
from qiskit.transpiler.basepasses import TransformationPass
from qiskit import transpile
from scipy.linalg import expm
import numpy as np
import math
from qiskit.synthesis import OneQubitEulerDecomposer


hls_config = HLSConfig(mcx=["2_dirty_kg24", "1_dirty_kg24", "noaux_hp24"], mcmt=["xgate", "noaux"])

np.random.seed(0)

J = 1
h = 1

zyz_decomposer = OneQubitEulerDecomposer(basis='ZYZ')

I_mat = np.array([[1, 0], [0, 1]])
X_mat = np.array([[0, 1], [1, 0]])
Y_mat = np.array([[0, -1j], [1j, 0]])
Z_mat = np.array([[1, 0], [0, -1]])
H_mat = 1 / math.sqrt(2) * np.array([[1, 1], [1, -1]])
Sdg_mat = (np.array([[1, 0],
                     [0, 0]]) + 1j *
           np.array([[0, 0],
                     [0, -1]]))
S_mat = (np.array([[1, 0],
                   [0, 0]]) + 1j *
         np.array([[0, 0],
                   [0, 1]]))
Tdg_mat = np.array([[1, 0],
                    [0, np.exp(-1j * np.pi / 4)]])
T_mat = np.array([[1, 0],
                  [0, np.exp(1j * np.pi / 4)]])

def synthesis_backend(qc):
    return transpile(qc, basis_gates=['cx', 's', 'sdg', 'h', 'u', 't', 'tdg'], optimization_level=0, hls_config=hls_config)

class CancelS4(TransformationPass):
    def run(self, dag):
        runs = dag.collect_1q_runs()
        for run in runs:
            idx = 0
            while idx + 3 < len(run):
                if all(node.op.name == "s" for node in run[idx:idx+4]):
                    for node in run[idx:idx+4]:
                        dag.remove_op_node(node)
                    idx += 4
                else:
                    idx += 1
        return dag

class CancelT2(TransformationPass):
    def run(self, dag):
        runs = dag.collect_1q_runs()
        for run in runs:
            idx = 0
            while idx + 1 < len(run):
                if all(node.op.name == "t" for node in run[idx:idx+2]):
                    dag.substitute_node(run[idx], SGate(), inplace=True)
                    dag.remove_op_node(run[idx+1])
                    idx += 2
                else:
                    idx += 1
        return dag

pass_manager = PassManager([InverseCancellation(), CancelT2(), CancelS4()])

def cliffordTbackend(qc):
    return pass_manager.run(transpile(qc, basis_gates=['cx', 's', 'sdg', 'h', 't', 'tdg'], optimization_level=0, unitary_synthesis_method="gridsynth"))

def count_gates(qc):
    cx_cnt = 0
    t_cnt = 0
    total_cnt = 0
    gate_set = set()

    for gate in qc.data:
        gate_set.add(gate.operation.name)
        if 'cx' == gate.operation.name:
            cx_cnt += 1
        if 't' == gate.operation.name or 'tdg' == gate.operation.name:
            t_cnt += 1
        total_cnt += 1

    print(gate_set)
    print("t count: ", t_cnt)
    print("cx count: ", cx_cnt)
    print("total count: ", total_cnt)
    print("depth: ", qc.depth())


def x(qc, idx):
    qc.h(idx)
    qc.s(idx)
    qc.s(idx)
    qc.h(idx)

def apply_control_states(qc, control_numbers, control_state):
    val = bin(control_state)[2:].zfill(control_numbers)

    for idx in range(len(val)):
        if val[idx] == '0':
            x(qc, qc.num_qubits - idx - 1)


def multicontrol_qiskit(qc_U, control_numbers, control_state):
    qubit_numbers = qc_U.num_qubits
    qc = QuantumCircuit(qubit_numbers + control_numbers)

    apply_control_states(qc, control_numbers, control_state)

    qc.compose(qc_U.control(control_numbers), [-1 - idx for idx in range(control_numbers)] + [idx for idx in range(qubit_numbers)], inplace=True)

    apply_control_states(qc, control_numbers, control_state)

    qc = synthesis_backend(qc)
    return qc