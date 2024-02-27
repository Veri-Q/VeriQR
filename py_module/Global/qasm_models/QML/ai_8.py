import cirq
import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X, Z, Y, Rxx, Power, BitFlipChannel, DepolarizingChannel, PhaseFlipChannel, RX, RZ, RY, U3
from mindquantum.io.qasm.openqasm import OpenQASM
import qlipschitz
from qlipschitz import calculate_lipschitz_, qasm2cirq_by_qiskit, noise_op, verification_
import csv
from random import choice

NUM_QUBITS = 8
WORKING_QUBITS = cirq.GridQubit.rect(1, NUM_QUBITS)

def generate_model_circuit_ai_8(variables):
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


def generate_model_circuit_for_ai_8(variables):
    qubits = [i for i in range(8)]
    variables = [round(i, 4) for i in variables]
    symbols = iter(variables)
    pi = np.pi
    circuit = Circuit()
    for q in qubits:
        # circuit += Power(Z, next(symbols)).on(q)
        circuit += RZ(next(symbols)).on(q)

    for q in qubits:
        # circuit += Power(Y, next(symbols)).on(q)
        circuit += RY(next(symbols)).on(q)

    for q in qubits:
        # circuit += Power(Z, next(symbols)).on(q)
        circuit += RZ(next(symbols)).on(q)

    for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]]):
        # circuit += XX(next(symbols)).on([q1, q2])
        circuit += Rxx(next(symbols)).on([q1, q2])

    for q in qubits:
        # circuit += Power(Z, next(symbols)).on(q)
        circuit += RZ(next(symbols)).on(q)

    for q in qubits:
        # circuit += Power(Y, next(symbols)).on(q)
        circuit += RY(next(symbols)).on(q)

    for q in qubits:
        # circuit += Power(Z, next(symbols)).on(q)
        circuit += RZ(next(symbols)).on(q)

    for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]]):
        # circuit += XX(next(symbols)).on([q1, q2])
        circuit += Rxx(next(symbols)).on([q1, q2])

    # circuit += Power(X, next(symbols)).on(qubits[-1])
    # circuit += Power(Y, next(symbols)).on(qubits[-1])
    # circuit += Power(X, next(symbols)).on(qubits[-1])
    circuit += RX(next(symbols)).on(qubits[-1])
    circuit += RZ(next(symbols)).on(qubits[-1])
    circuit += RX(next(symbols)).on(qubits[-1])

    circuit_str = OpenQASM().to_string(circuit)
    f = open('./ai_8.qasm', 'w')
    f.write(circuit_str)
    f.close()
    # if p > 1e-5:
    #     if mixed:
    #         for q in qubits[::3]:
    #             circuit += BitFlipChannel(p).on(q)
    #         for q in qubits[1::3]:
    #             circuit += DepolarizingChannel(p).on(q)
    #         for q in qubits[2::3]:
    #             circuit += PhaseFlipChannel(p).on(q)
    #     else:
    #         for q in qubits:
    #             circuit += noise_op(p).on(q)
    # return circuit


# imported from a trained model
params_ai_8 = [5.6864963,  4.5384674,  4.0658937,  6.0114822,  2.6314237,  0.7971049,
           6.2414956,  1.231465 ,  5.112798  , 0.09745377, 0.2654334,  4.1310773,
           3.3447504,  5.935498 ,  1.7449    , 1.745954  , 1.514159 ,  2.4577525,
           6.188601 ,  5.751889 ,  0.16371164, 5.015923  , 2.698336 ,  2.7948823,
           1.7905817,  4.1858573,  1.714581  , 4.134787  , 4.522799 ,  0.33325404,
           5.646758 ,  1.0231644,  3.535049  , 4.513359  , 2.4423301,  3.346549,
           0.7184883,  3.5541363,  5.1378045 , 5.4350505 , 4.250444 ,  2.081229,
           2.3359709,  1.1259285,  3.906016  , 0.1284471 , 2.5366719,  5.801898,
           1.9489733,  2.5943935,  5.240497  , 2.2280385 , 2.2115154,  3.0721598,
           0.9330431,  2.9257228,  2.702144  , 4.1977177 , 1.682387 ,  3.859512,
           4.688113 ,  5.4294186,  3.3565576 , 6.080049  , 1.753433 ,  1.5129646,
           5.4340334, ]

# generate_model_circuit_for_ai_8(params_ai_8)

circuit_cirq = generate_model_circuit_ai_8(params_ai_8)
# calculate_lipschitz_(circuit_cirq, WORKING_QUBITS)

# qubits, circuit_cirq, qasm_str = qasm2cirq_by_qiskit('./ai_8.qasm')
# calculate_lipschitz_(circuit_cirq, qubits)


# def fileTest(model_name, qubits):
#     with open("../../results/{}}.csv".format(model_name), "a+") as csvfile:
#         w = csv.writer(csvfile)
#         for noise_type in ["bit_flip", "depolarizing", "phase_flip", "mixed"]:
#             noise_op_ = noise_op[noise_type]
#             for p in [0.01, 0.001]:
#                 circuit_cirq_ = circuit_cirq
#                 # add noise
#                 if p > 1e-7:
#                     if noise_type == "mixed":
#                         circuit_cirq_ += cirq.bit_flip(p).on_each(*qubits[::3])
#                         circuit_cirq_ += cirq.depolarize(p).on_each(*qubits[1::3])
#                         circuit_cirq_ += cirq.phase_flip(p).on_each(*qubits[2::3])
#                     else:
#                         circuit_cirq_ += noise_op_(p).on_each(*qubits)
#
#                 k, total_time = calculate_lipschitz_(circuit_cirq_, qubits)
#                 print('Noise configuration: {}, {}\n'.format(noise_type, p))
#
#                 # 逐行写入数据 (写入多行用writerows)
#                 w.writerow([noise_type, p, np.round(k, 5), np.round(total_time, 2)])

def fileTest(model_name, qubits):
    with open("../../results/global_results.csv", "a+") as csvfile:
        w = csv.writer(csvfile)
        for noise_type in ["bit_flip", "depolarizing", "phase_flip", "mixed"]:
            noise_op_ = noise_op[noise_type]
            p = choice([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075])
            epsilon = choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075])
            delta = choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.0075])
            circuit_cirq_ = circuit_cirq
            # add noise
            if p > 1e-7:
                if noise_type == "mixed":
                    circuit_cirq_ += cirq.bit_flip(p).on_each(*qubits[::3])
                    circuit_cirq_ += cirq.depolarize(p).on_each(*qubits[1::3])
                    circuit_cirq_ += cirq.phase_flip(p).on_each(*qubits[2::3])
                else:
                    circuit_cirq_ += noise_op_(p).on_each(*qubits)

            # k, total_time = calculate_lipschitz_(circuit_cirq_, qubits)
            flag, k, bias_kernel, total_time = verification_(circuit_cirq_, qubits, epsilon, delta)
            # print(flag)
            # print(k)
            # print(bias_kernel)
            # print(total_time)

            print('Noise configuration: {}, {}\n'.format(noise_type, p))

            res = 'YES' if flag else 'NO'
            w.writerow([model_name, noise_type.replace('_', ' '), p, (epsilon, delta),
                        '%.5f' % k, res, '%.2f' % total_time])
            # 逐行写入数据 (写入多行用writerows)
            # w.writerow([model_name, noise_type.replace('_',' '), p, '%.5f' % k, '%.2f' % total_time])


# fileTest('ai_8', WORKING_QUBITS)
