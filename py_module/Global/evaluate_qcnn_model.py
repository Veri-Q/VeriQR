from qlipschitz import lipschitz
import time
import cirq
import numpy as np
import sys

## Quantum Convolutional Neural Network ...

def one_qubit_unitary(bit, variables):
    """Make a Cirq circuit enacting a rotation of the bloch sphere about the X,
    Y and Z axis, that depends on the values in `symbols`.
    """
    return cirq.Circuit(
        cirq.X(bit)**next(variables),
        cirq.Y(bit)**next(variables),
        cirq.Z(bit)**next(variables))


def two_qubit_unitary(bits, variables):
    """Make a Cirq circuit that creates an arbitrary two qubit unitary."""
    circuit = cirq.Circuit()
    circuit += one_qubit_unitary(bits[0], variables)
    circuit += one_qubit_unitary(bits[1], variables)
    circuit += [cirq.ZZ(*bits)**next(variables)]
    circuit += [cirq.YY(*bits)**next(variables)]
    circuit += [cirq.XX(*bits)**next(variables)]
    circuit += one_qubit_unitary(bits[0], variables)
    circuit += one_qubit_unitary(bits[1], variables)
    return circuit


def two_qubit_pool(source_qubit, sink_qubit, variables):
    """Make a Cirq circuit to do a parameterized 'pooling' operation, which
    attempts to reduce entanglement down from two qubits to just one."""
    pool_circuit = cirq.Circuit()
    sink_basis_selector = one_qubit_unitary(sink_qubit, variables)
    source_basis_selector = one_qubit_unitary(source_qubit, variables)
    pool_circuit.append(sink_basis_selector)
    pool_circuit.append(source_basis_selector)
    pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
    pool_circuit.append(sink_basis_selector**-1)
    return pool_circuit

def quantum_conv_circuit(bits, variables):
    """Quantum Convolution Layer.
    Return a Cirq circuit with the cascade of `two_qubit_unitary` applied
    to all pairs of qubits in `bits` as in the diagram above.
    """
    circuit = cirq.Circuit()
    for first, second in zip(bits[0::2], bits[1::2]):
        circuit += two_qubit_unitary([first, second], variables)
    for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
        circuit += two_qubit_unitary([first, second], variables)
    return circuit

def quantum_pool_circuit(source_bits, sink_bits, variables):
    """A layer that specifies a quantum pooling operation.
    A Quantum pool tries to learn to pool the relevant information from two
    qubits onto 1.
    """
    circuit = cirq.Circuit()
    for source, sink in zip(source_bits, sink_bits):
        circuit += two_qubit_pool(source, sink, variables)
    return circuit

def quantum_full_circuit(qubits, variables):
    """Quantum Full-connect Layer"""
    circuit = cirq.Circuit()
    
    circuit += [cirq.X(q) ** next(variables) for q in qubits]
    circuit += [cirq.Y(q) ** next(variables) for q in qubits]
    circuit += [cirq.X(q) ** next(variables) for q in qubits]
    
    if len(qubits) >= 2:
        circuit += [cirq.XX(q1, q2) ** next(variables) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    
    circuit += [cirq.X(qubits[-1]) ** next(variables), cirq.Y(qubits[-1]) ** next(variables), cirq.X(qubits[-1]) ** next(variables)]
    
    return circuit
    

def create_model_circuit(qubits, variables, p=0.0, noise_op=cirq.depolarize, mixed=False, full_size=4):
    """Create sequence of alternating convolution and pooling operators 
    which gradually shrink over time."""
    
    qnum = len(qubits)
    if qnum <= full_size:
        return quantum_full_circuit(qubits, variables)
    
    circuit = cirq.Circuit()
    circuit += quantum_conv_circuit(qubits, variables)
    if p > 1e-5:
        if mixed:
            circuit += cirq.bit_flip(p).on_each(*qubits[::3])
            circuit += cirq.depolarize(p).on_each(*qubits[1::3])
            circuit += cirq.phase_flip(p).on_each(*qubits[2::3])
        else:
            circuit += noise_op(p).on_each(*qubits)

    circuit += create_model_circuit(qubits[qnum//2:], variables, p=p, noise_op=noise_op, full_size=full_size)
    
    return circuit

def random_variables():
    while True:
        yield np.random.rand()

if __name__ == '__main__':
    QUBITS_NUM = int(sys.argv[1])
    NOISE_TYPE = str(sys.argv[2])
    noise_op = {
        "phase_flip": cirq.phase_flip,
        "depolarize": cirq.depolarize,
        "bit_flip": cirq.bit_flip,
        "mixed": cirq.depolarize
    }
    if NOISE_TYPE == "mixed":
        mixed = True
    else:
        mixed = False

    QUBITS = cirq.GridQubit.rect(1, QUBITS_NUM)
    VARS = random_variables()
    N = 100
    p = [1e-4] * 3 + [1e-3] * 3 + [1e-2] * 3
    for noisy_p in p:
        tstart = time.time()
        #print("\n===============================================")
        print(f"\n==================p = {noisy_p:.4f}===================")
        model_circuit = create_model_circuit(QUBITS, VARS, noisy_p, full_size=QUBITS_NUM//2+1, noise_op=noise_op[NOISE_TYPE], mixed=mixed)
        k = lipschitz(model_circuit, QUBITS, np.array([[1.,0.],[0.,0.]]))
        if k != -1:
            print("Lipschitz K = ", k)
        else:
            print("Lipschitz K = -")

        print(f"Elapsed time = {(time.time() - tstart)/60:.4f}m")