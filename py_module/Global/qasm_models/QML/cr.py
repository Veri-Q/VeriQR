import cirq
from baseline import fileTest

NUM_QUBITS = 9
WORKING_QUBITS = cirq.GridQubit.rect(1, NUM_QUBITS)


def generate_model_circuit(variables):
    qubits = WORKING_QUBITS
    symbols = iter(variables)
    circuit = cirq.Circuit()
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]

    circuit += [cirq.XX(q1, q2) ** next(symbols) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.XX(q1, q2) ** next(symbols) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]

    circuit += cirq.X(qubits[-1]) ** next(symbols)
    circuit += cirq.Y(qubits[-1]) ** next(symbols)
    circuit += cirq.X(qubits[-1]) ** next(symbols)

    return circuit


# imported from a trained model
params = [0.25005275, 0.1326074, 2.7016566, 4.0975943, 3.4119093, 4.7883525,
          2.724837, 3.5035949, 5.2497797, 3.109118, 1.2774808, 4.773297,
          2.529881, 6.0996304, 2.6149085, 0.36178315, 1.7953006, 3.7028892,
          5.8374367, 5.6633015, 1.1470748, 0.38180125, 0.45548266, 4.427361,
          5.7957864, 3.8033733, 5.544324, 6.214601, 5.1286793, 1.4030437,
          3.8545115, 0.18550971, 3.1386983, 5.195878, 1.4268013, 4.9385486,
          0.26109523, 1.3583922, 1.1371251, 5.902646, 1.9806675, 3.643112,
          0.4566148, 1.2792462, 3.2569685, 0.30495742, 1.3238809, 4.924705,
          3.2494488, 3.6414773, 4.4106135, 3.9998493, 4.9263744, 5.0355964,
          1.8715085, 2.821595, 3.9963686, 3.9364269, 2.0835845, 5.255737,
          5.682012, 1.5800779, 0.13781616, 4.6939526, 1.7238493, 2.7945642,
          1.7180046, 3.3467932, 2.6824343, 1.4030236, 4.50502, 6.0017667,
          2.029843, 2.7211223, 5.819587, ]

circuit_cirq = generate_model_circuit(params)

fileTest('gc_9', WORKING_QUBITS)
