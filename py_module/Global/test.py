import mindquantum as mq
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import CNOT, H, X, Z, Y, Rxx, Power, BitFlipChannel, DepolarizingChannel, PhaseFlipChannel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import gc
import time
from qlipschitz import *


def calculate_lipschitz_(model_circuit, qubits):
    measurement = np.array([[1., 0.], [0., 0.]])

    print("===========The Lipschitz Constant Calculation Start============")
    start_time = time.time()
    k, bias_kernel = lipschitz(model_circuit, qubits, measurement)
    total_time = time.time() - start_time

    print('Lipschitz K =', k)
    print('Elapsed time = %.4fs' % total_time)
    # print('The bias kernel is: (\n{},\n {})'.format(bias_kernel[0], bias_kernel[1]))
    print("============The Lipschitz Constant Calculation End=============")
    return k, total_time, bias_kernel


def verification_(model_circuit, qubits, epsilon, delta):
    k, total_time, bias_kernel = calculate_lipschitz_(model_circuit, qubits)
    start = time.time()
    if delta >= k * epsilon:
        return True, k, [], total_time + time.time() - start
    return False, k, bias_kernel, total_time + time.time() - start


def batchTest(path):
    files = os.listdir(path)
    for file in files:
        fileTest(path + file)


def fileTest(file):
    qubits, circuit_cirq, qasm_str = qasm2cirq_by_qiskit(file)
    circuit_mq = qasm2mq(qasm_str)

    all_measures = []
    for gate in circuit_mq:
        if type(gate) is Measure:
            all_measures.append(gate)

    model_name = file[file.rfind("/") + 1: file.index(".qasm")]

    # with open("./results/{}.csv".format(model_name), "a+") as csvfile:
    with (open("./results/global_results.csv", "a+") as csvfile):
        w = csv.writer(csvfile)
        for noise_type in ["bit_flip", "depolarizing", "phase_flip", "mixed"]:
            # for noise_type in ["mixed"]:
            noise_op_ = noise_op[noise_type]
            noise_op_mq_ = noise_op_mq[noise_type]
            p = choice([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075])
            epsilon = choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075])
            delta = choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.0075])
            # circuit_cirq_ = cirq.circuits.Circuit()
            # circuit_cirq_ += circuit_cirq
            circuit_cirq_ = circuit_cirq
            # circuit_mq_ = Circuit()
            # circuit_mq_ += circuit_mq
            circuit_mq_ = circuit_mq
            if circuit_mq_.has_measure_gate:
                circuit_mq_ = circuit_mq_.remove_measure()
            # add noise
            if p > 1e-7:
                if noise_type == "mixed":
                    circuit_cirq_ += cirq.bit_flip(p).on_each(*qubits[::3])
                    circuit_cirq_ += cirq.depolarize(p).on_each(*qubits[1::3])
                    circuit_cirq_ += cirq.phase_flip(p).on_each(*qubits[2::3])
                    n_qubits = range(circuit_mq_.n_qubits)
                    for q in n_qubits[::3]:
                        circuit_mq_ += BitFlipChannel(p).on(q)
                    for q in n_qubits[1::3]:
                        circuit_mq_ += DepolarizingChannel(p).on(q)
                    for q in n_qubits[2::3]:
                        circuit_mq_ += PhaseFlipChannel(p).on(q)
                else:
                    circuit_cirq_ += noise_op_(p).on_each(*qubits)
                    for q in range(circuit_mq_.n_qubits):
                        circuit_mq_ += noise_op_mq_(p).on(q)

            for m in all_measures:
                circuit_mq_ += m
            # file_name = model_name + "_" + noise_type + "_" + str(p)
            # print("file_name: {}".format(file_name))
            # circuit_mq_.svg().to_file("./model_circuits/circuit_{}.svg".format(file_name))
            # print("===========Printing Model Circuit End============")

            # k, total_time = calculate_lipschitz_(circuit_cirq_, qubits)
            flag, k, bias_kernel, total_time = verification_(circuit_cirq_, qubits, epsilon, delta)
            print('Circuit: %s' % file)
            print('Noise configuration: {}, {}\n'.format(noise_type, p))
            res = 'YES' if flag else 'NO'
            w.writerow([model_name, noise_type.replace('_', ' '), p, (epsilon, delta),
                        '%.5f' % k, res, '%.2f' % total_time])

        gc.collect()


# batchTest('./qasm_models/HFVQE/')

# fileTest('./qasm_models/HFVQE/hf_6_0_5.qasm')
# fileTest('./qasm_models/HFVQE/hf_8_0_5.qasm')
# fileTest('./qasm_models/HFVQE/hf_10_0_5.qasm')
# fileTest('./qasm_models/HFVQE/hf_12_0_5.qasm')
# fileTest('./qasm_models/QAOA/qaoa_10.qasm')
# fileTest('./qasm_models/inst/inst_4x4_10_0.qasm')

# fileTest('./qasm_models/fashion.qasm')
# fileTest('./qasm_models/iris.qasm')

# for d0 in range(1, 10):
#     for d1 in range(d0+1, 10):
#         digits = str(d0) + str(d1)
#         fileTest('./qasm_models/mnist{}.qasm'.format(digits))
