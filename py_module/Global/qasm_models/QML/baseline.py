import cirq
import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X, Z, Y, Rxx, Power, BitFlipChannel, DepolarizingChannel, PhaseFlipChannel, RX, RZ, \
    RY, U3, Measure
from mindquantum.io.qasm.openqasm import OpenQASM
# import qlipschitz
# from qlipschitz import calculate_lipschitz_, qasm2cirq_by_qiskit, noise_op, verification_
import sys
import csv
from random import choice, uniform
from numpy import load
import time
from math import sqrt
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers
from cirq.contrib.qasm_import import circuit_from_qasm

noise_op_cirq = {
    "phase_flip": cirq.phase_flip,
    "depolarizing": cirq.depolarize,
    "bit_flip": cirq.bit_flip,
    "mixed": cirq.depolarize
}

noise_op_mq = {
    "phase_flip": PhaseFlipChannel,
    "depolarizing": DepolarizingChannel,
    "bit_flip": BitFlipChannel,
    "mixed": DepolarizingChannel
}

noise_ops = ["phase_flip", "depolarizing", "bit_flip"]

params_cr = [0.25005275, 0.1326074, 2.7016566, 4.0975943, 3.4119093, 4.7883525,
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

params_aci = [5.6864963, 4.5384674, 4.0658937, 6.0114822, 2.6314237, 0.7971049,
              6.2414956, 1.231465, 5.112798, 0.09745377, 0.2654334, 4.1310773,
              3.3447504, 5.935498, 1.7449, 1.745954, 1.514159, 2.4577525,
              6.188601, 5.751889, 0.16371164, 5.015923, 2.698336, 2.7948823,
              1.7905817, 4.1858573, 1.714581, 4.134787, 4.522799, 0.33325404,
              5.646758, 1.0231644, 3.535049, 4.513359, 2.4423301, 3.346549,
              0.7184883, 3.5541363, 5.1378045, 5.4350505, 4.250444, 2.081229,
              2.3359709, 1.1259285, 3.906016, 0.1284471, 2.5366719, 5.801898,
              1.9489733, 2.5943935, 5.240497, 2.2280385, 2.2115154, 3.0721598,
              0.9330431, 2.9257228, 2.702144, 4.1977177, 1.682387, 3.859512,
              4.688113, 5.4294186, 3.3565576, 6.080049, 1.753433, 1.5129646,
              5.4340334, ]

params_fct = [0.11780868, 1.5765338, 4.206496, 0.5947907, 6.0406756, 3.2344778,
              2.0535638, 1.0474278, 1.3552234, 1.1947954, 4.359093, 4.3828235,
              1.5595611, 4.189004, 4.736576, 5.6395154, 5.4876723, 3.7906342,
              0.896061, 5.0224333, 4.600445, 5.46947, 2.2689416, 1.4538898,
              2.2451863, 3.6725183, 1.8202529, 1.6112416, 0.574555, 4.0879498,
              5.6109347, 3.6359, 6.2621737, 4.9480653, 2.7919254, 5.074803,
              5.822844, 5.5694394, 5.677946, 5.1136017, 1.9180884, 2.2606523,
              3.8960311, 5.540094, 1.9288703, 4.161004, 5.011807, 1.5809758,
              1.9225371, 0.47577053, 5.9932785, 6.2445574, 0.36193165, 0.54220635,
              2.5442297, 6.1613083, 2.1198325, 5.00303, 0.99314445, 3.1671383,
              1.9087403, 0.6342722, 0.70649546, 3.2471435, 3.4544551, 3.4269898,
              5.728249, 1.6742734, 3.6606266, 1.8093376, 1.574797, 6.1125684,
              5.2926126, 0.16639477, 5.572203, ]

case_params = {
    'aci': (params_aci, 8),
    'cr': (params_cr, 9),
    'fct': (params_fct, 9),
}

model_name = str(sys.argv[1])
if '.qasm' in model_name:
    model_name = model_name[model_name.rfind('/')+1:-5]
else:
    variables, qubits_num = case_params[model_name]
arg_num = len(sys.argv)
noise_list = []
kraus_file = ''
if arg_num <= 2:  # random noise
    noise_type = choice(noise_ops)
    noise = noise_op_mq[noise_type].__name__
    noise = noise[0: noise.index("Channel")]
    noisy_p = float(round(uniform(0, 0.2), 5))  # 随机数的精度round(数值，精度)
    file_name = "{}_{}_{}".format(model_name, noise, str(noisy_p))
else:
    noise_type = str(sys.argv[2])
    noisy_p = float(sys.argv[arg_num - 3])
    if noise_type == 'mixed':
        noise_list = [i for i in sys.argv[3: arg_num - 3]]
        noise_list_ = [noise_op_mq[i].__name__ for i in noise_list]
        noise_list_ = [i[0: i.index("Channel")] for i in noise_list_]
        print("noise_list: ", noise_list)
        file_name = "{}_mixed_{}_{}".format(model_name, '_'.join(noise_list_), str(noisy_p))
    elif noise_type == 'custom':
        kraus_file = sys.argv[3]
        file_name = "{}_custom_{}_{}".format(model_name, kraus_file[kraus_file.rfind('/') + 1:-4], str(noisy_p))
    else:
        noise = noise_op_mq[noise_type].__name__
        noise = noise[0: noise.index("Channel")]
        file_name = "{}_{}_{}".format(model_name, noise, str(noisy_p))

epsilon = float(sys.argv[arg_num - 2])
delta = float(sys.argv[arg_num - 1])


def generate_model_circuit(variables, qubits_num):
    qubits = cirq.GridQubit.rect(1, qubits_num)
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

    # if p > 1e-5:
    #     if noise_type == "mixed":
    #         l = len(noise_list)
    #         for q in range(qubits_num)[::l]:
    #             for i in range(l):
    #                 if q + i < qubits_num:
    #                     circuit += noise_op_cirq[noise_list[i]](p)(qubits[q + i])
    #     elif noise_type == "custom":
    #         # TODO
    #         data = load(kraus_file)
    #         noisy_kraus = data['kraus']
    #     else:
    #         # noise = noise_op_cirq[noise_type]
    #         circuit += noise_op_cirq[noise_type](p).on_each(*qubits)

    return qubits, circuit


def print_model_circuit(file_name, variables, qubits_num, noise_type, noise_list, kraus_file, p):
    qubits = [i for i in range(qubits_num)]
    variables = [round(i, 4) for i in variables]
    symbols = iter(variables)
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

    if noise_type == "mixed":
        l = len(noise_list)
        for q in qubits[::l]:
            for i in range(l):
                if q + i < qubits_num:
                    circuit += noise_op_mq[noise_list[i]](p).on(q + i)
    elif noise_type == "custom":
        # TODO
        data = load(kraus_file)
        noisy_kraus = data['kraus']
    else:
        for q in qubits:
            circuit += noise_op_mq[noise_type](p).on(q)

    circuit += Measure('q{}'.format(qubits[-1])).on(qubits[-1])
    circuit.svg().to_file("./figures/{}.svg".format(file_name))


def circuit2M(qubits, circuit, noise_type, noise_list, kraus_file, p):
    U1 = cirq.unitary(circuit)
    qubits_num = len(qubits)
    if p > 1e-5:
        noisy_kraus = []
        if noise_type == "mixed":
            l = len(noise_list)
            for q in range(qubits_num)[::l]:
                for i in range(l):
                    if q + i < qubits_num:
                        # noisy_kraus += cirq.kraus(noise_op_cirq[noise_list[i]](p)(qubits[q + i]))
                        kraus = noise_op_cirq[noise_list[i]](p)(qubits[q + i])._mixture_()
                        kraus_ = []
                        for E in kraus:
                            # print()
                            kraus_.append(sqrt(E[0]) * E[1])
                        noisy_kraus.append(kraus_)
        elif noise_type == "custom":
            # TODO
            data = load(kraus_file)
            noisy_kraus = data['kraus']
        else:
            # noise = noise_op_cirq[noise_type]
            # noisy_kraus = [cirq.kraus(noise_op_cirq[noise_type](p)(q)) for q in qubits]
            for q in qubits:
                kraus = noise_op_cirq[noise_type](p)(q)._mixture_()
                kraus_ = []
                for E in kraus:
                    # print()
                    kraus_.append(sqrt(E[0]) * E[1])
                noisy_kraus.append(kraus_)

    M = U1.conj().T @ np.kron(np.eye(2 ** (qubits_num - 1)), np.array([[1., 0.], [0., 0.]])) @ U1
    # print("M:", M.shape)

    # if p > 1e-5:
    #     for j in range(qubits_num):
    #         N = 0
    #         for E in noisy_kraus[j]:
    #             print("E:", E)
    #             F = np.kron(np.eye(2 ** j), np.kron(E, np.eye(2 ** (qubits_num - j - 1))))
    #             print("F:", F.shape)
    #             N = F.conj().T @ M @ F + N
    #
    #         M = N
    if p > 1e-5:
        for j in range(qubits_num):
            N = 0
            for E in noisy_kraus[j]:
                # print("E:", E)
                F = np.kron(np.eye(2 ** j), np.kron(E, np.eye(2 ** (qubits_num - j - 1))))
                # print("F:", F.shape)
                N += F.conj().T @ M @ F
            M = N

    M = U1.conj().T @ M @ U1
    return M


def qasm2cirq(file):
    with open(file, 'r') as f:
        qasm_str = f.read()

    # qasm to qiskit
    circ = QuantumCircuit.from_qasm_str(qasm_str)
    circ.remove_final_measurements()
    circ = RemoveBarriers()(circ)
    # qiskit to qasm
    qasm_str = circ.inverse().qasm()

    # qasm to cirq
    circ = circuit_from_qasm(qasm_str)
    qubits = sorted(circ.all_qubits())

    return qubits, circ


if '.qasm' in str(sys.argv[1]):
    qubits, circuit = qasm2cirq(str(sys.argv[1]))
else:
    qubits, circuit = generate_model_circuit(variables, qubits_num)
# generate_model_circuit(variables)

t_start = time.time()
print("\n===========The Lipschitz Constant Calculation Start============")
a, _ = np.linalg.eig(circuit2M(qubits, circuit, noise_type, noise_list, kraus_file, noisy_p))
k = np.real(max(a) - min(a))
total_time = time.time() - t_start

with (open("../../results/global_results.csv", "a+") as csvfile):
    w = csv.writer(csvfile)
    # for epsilon, delta in [(0.003, 0.0001), (0.03, 0.0005),]
    # epsilon, delta = 0.003, 0.0001
    # epsilon, delta = 0.03, 0.0005
    # epsilon, delta = 0.05, 0.001
    # epsilon, delta = 0.005, 0.005

    # epsilon, delta = 0.075, 0.003
    # epsilon, delta = 0.0003, 0.0001
    # epsilon, delta = 0.01, 0.0075
    # epsilon, delta = 0.075, 0.0075

    # epsilon, delta = 0.01, 0.0005
    # epsilon, delta = 0.075, 0.005
    # epsilon, delta = 0.0003, 0.0001
    # epsilon, delta = 0.0001, 0.0001
    start = time.time()
    res = 'YES' if delta >= k * epsilon else 'NO'
    total_time += time.time() - start
    w.writerow([model_name, noise_type.replace('_', ' '), noisy_p, (epsilon, delta),
                '%.5f' % k, res, '%.2f' % total_time])

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

# def fileTest(model_name, qubits):
#     with open("../../results/global_results.csv", "a+") as csvfile:
#         w = csv.writer(csvfile)
#         for noise_type in ["bit_flip", "depolarizing", "phase_flip", "mixed"]:
#             noise_op_ = noise_op[noise_type]
#             p = choice([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075])
#             epsilon = choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075])
#             delta = choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.0075])
#             circuit_cirq_ = circuit_cirq
#             # add noise
#             if p > 1e-7:
#                 if noise_type == "mixed":
#                     circuit_cirq_ += cirq.bit_flip(p).on_each(*qubits[::3])
#                     circuit_cirq_ += cirq.depolarize(p).on_each(*qubits[1::3])
#                     circuit_cirq_ += cirq.phase_flip(p).on_each(*qubits[2::3])
#                 else:
#                     circuit_cirq_ += noise_op_(p).on_each(*qubits)
#
#             # k, total_time = calculate_lipschitz_(circuit_cirq_, qubits)
#             flag, k, bias_kernel, total_time = verification_(circuit_cirq_, qubits, epsilon, delta)
#             # print(flag)
#             # print(k)
#             # print(bias_kernel)
#             # print(total_time)
#
#             print('Noise configuration: {}, {}\n'.format(noise_type, p))
#
#             res = 'YES' if flag else 'NO'
#             w.writerow([model_name, noise_type.replace('_', ' '), p, (epsilon, delta),
#                         '%.5f' % k, res, '%.2f' % total_time])
#             # 逐行写入数据 (写入多行用writerows)
#             # w.writerow([model_name, noise_type.replace('_',' '), p, '%.5f' % k, '%.2f' % total_time])
#
# # fileTest('ai_8', WORKING_QUBITS)
