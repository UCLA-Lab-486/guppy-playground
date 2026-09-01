"""
Demo mirroring the Qiskit qvst_main.py: simulate the 2-qubit Heisenberg
Hamiltonian via qubitization and compare against the exact evolution.

Run with:
    venv/bin/python -m qubitization.main
"""

import numpy as np
from scipy.linalg import expm

from qubitization.util import Mat
from qubitization.params import num_qubits, time_t, J, h
from qubitization.qvst import qvst_algo
from qubitization.block_encodings import qvst_heis_chain
from qubitization.test_qvst import extract_block

circuit, total, control_numbers = qvst_algo(num_qubits, time_t, qvst_heis_chain)

result_unitary = extract_block(circuit, total, num_qubits)

Hermitian_mat = (J * (np.kron(Mat.Z, Mat.Z) + np.kron(Mat.X, Mat.X) + np.kron(Mat.Y, Mat.Y)) +
                 h * (np.kron(Mat.X, Mat.I) + np.kron(Mat.I, Mat.X)))

target_unitary = expm(-1j * Hermitian_mat / (J * 3 * (num_qubits - 1) + h * num_qubits) * time_t)

print(np.diag(result_unitary @ np.conjugate(target_unitary).T))
print(np.abs(np.diag(result_unitary @ np.conjugate(target_unitary).T)))
