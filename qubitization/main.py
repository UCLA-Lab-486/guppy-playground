import numpy as np
from util import Mat
from scipy.linalg import expm
from qubitization.params import num_qubits, time_t, J, h


Hermitian_mat = (J * (np.kron(Mat.Z, Mat.Z) + np.kron(Mat.X, Mat.X) + np.kron(Mat.Y, Mat.Y)) +
                 h * (np.kron(Mat.X, Mat.I) + np.kron(Mat.I, Mat.X)))

target_unitary = expm(-1j * Hermitian_mat / (J * 3 * (num_qubits-1) + h * num_qubits) * time_t)

print(np.diag(result_unitary @ np.conjugate(target_unitary).T))
print(np.abs(np.diag(result_unitary @ np.conjugate(target_unitary).T)))