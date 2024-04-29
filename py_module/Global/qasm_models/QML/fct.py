import cirq
from baseline import fileTest

NUM_QUBITS = 9
WORKING_QUBITS = cirq.GridQubit.rect(1,NUM_QUBITS)

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
params = [ 0.11780868, 1.5765338 , 4.206496  , 0.5947907 , 6.0406756 , 3.2344778,
           2.0535638 , 1.0474278 , 1.3552234 , 1.1947954 , 4.359093  , 4.3828235,
           1.5595611 , 4.189004  , 4.736576  , 5.6395154 , 5.4876723 , 3.7906342,
           0.896061  , 5.0224333 , 4.600445  , 5.46947   , 2.2689416 , 1.4538898,
           2.2451863 , 3.6725183 , 1.8202529 , 1.6112416 , 0.574555  , 4.0879498,
           5.6109347 , 3.6359    , 6.2621737 , 4.9480653 , 2.7919254 , 5.074803,
           5.822844  , 5.5694394 , 5.677946  , 5.1136017 , 1.9180884 , 2.2606523,
           3.8960311 , 5.540094  , 1.9288703 , 4.161004  , 5.011807  , 1.5809758,
           1.9225371 , 0.47577053, 5.9932785 , 6.2445574 , 0.36193165, 0.54220635,
           2.5442297 , 6.1613083 , 2.1198325 , 5.00303   , 0.99314445, 3.1671383,
           1.9087403 , 0.6342722 , 0.70649546, 3.2471435 , 3.4544551 , 3.4269898,
           5.728249  , 1.6742734 , 3.6606266 , 1.8093376 , 1.574797  , 6.1125684,
           5.2926126 , 0.16639477, 5.572203  , ]

circuit_cirq = generate_model_circuit(params)

fileTest('ec_9', WORKING_QUBITS)