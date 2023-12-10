import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import numpy as np
import time
import sys

def circuit2M(p, variables, noise_op=cirq.depolarize, mixed=False):
    qubits = cirq.GridQubit.rect(1,9)
    num = len(qubits)
    variables = iter(variables)
    circuit = cirq.Circuit()
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    U1 = cirq.unitary(circuit)

    if p > 1e-5:
        if mixed:
            noisy_kraus = [cirq.channel(cirq.bit_flip(p)(q)) for q in qubits[::3]]
            noisy_kraus += [cirq.channel(cirq.depolarize(p)(q)) for q in qubits[1::3]]
            noisy_kraus += [cirq.channel(cirq.phase_flip(p)(q)) for q in qubits[2::3]]
        else:
            noisy_kraus = [cirq.channel(noise_op(p)(q)) for q in qubits] 
    
    circuit = cirq.Circuit()
    circuit += [cirq.XX(q1, q2) ** next(variables) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.XX(q1, q2) ** next(variables) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    U2 = cirq.unitary(circuit)

    M = U2.conj().T @ np.kron(np.eye(2 ** (num - 1)), np.array([[1.,0.], [0.,0.]])) @ U2
    
    if p > 1e-5:
        for j in range(num):
            N = 0
            for E in noisy_kraus[j]:
                F = np.kron(np.eye(2 ** j), np.kron(E, np.eye(2 ** (num-j-1))))
                N = F.conj().T @ M @ F + N
        
            M = N
    
    M = U1.conj().T @ M @ U1
    return M

noise_type = str(sys.argv[1])
noisy_p = float(sys.argv[2])
noise_op = {
        "phase_flip": cirq.phase_flip,
        "depolarize": cirq.depolarize,
        "bit_flip": cirq.bit_flip,
        "mixed": cirq.depolarize
}
if noise_type == "mixed":
    mixed = True
else:
    mixed = False

print("=================Loading Start=================")
noisy_model = tf.keras.models.load_model(f'saved_model/gc_{noise_type}_{noisy_p}')
print("==================Loading End==================")

tstart = time.time()
print("\n===========Lipschitz Constant Start============")
a,_= np.linalg.eig(circuit2M(noisy_p,noisy_model.layers[1].get_weights()[0],noise_op[noise_type], mixed))
k = np.real(max(a) - min(a))
if k != -1:
    print("Lipschitz K = ", k)
else:
    print("Lipschitz K = -")
print(f"Elapsed time = {(time.time() - tstart):.4f}s")
print("============Lipschitz Constant End=============")
