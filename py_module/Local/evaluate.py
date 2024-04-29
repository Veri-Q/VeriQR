from VeriQ import RobustnessVerifier, PureRobustnessVerifier
from prettytable import PrettyTable
from sys import argv
import numpy as np
from numpy import load

from mindquantum.io import OpenQASM
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import BitFlipChannel, DepolarizingChannel, PhaseFlipChannel, Measure, I
from mindquantum.core.gates.basic import BasicGate
import random
import mindquantum
import gc
import csv
from multiprocessing import Pool
import copy

noise_ops = ["phase_flip", "depolarizing", "bit_flip"]

noise_op_map = {
    "bit_flip": BitFlipChannel,
    "depolarizing": DepolarizingChannel,
    "phase_flip": PhaseFlipChannel,
}

I_matrix = mindquantum.core.gates.I.matrix()


def qasm2mq(qasm_file):
    f = open(qasm_file)
    qasm = f.read()
    f.close()
    circuit = OpenQASM().from_string(qasm)
    if circuit.parameterized:
        val_list = []
        for param in circuit.params_name:
            param = param.replace('pi', str(np.pi)).replace('π', str(np.pi))
            val_list.append(float(param))
        pr = dict(zip(circuit.params_name, val_list))  # 获取线路参数
        circuit = circuit.apply_value(pr)

    # model_name = "{}_origin.svg".format(qasm_file[qasm_file.rfind('/') + 1:-5])
    # circuit.svg().to_file("./figures/" + model_name)  # qasm_file chop '.qasm'
    # print(model_name + " saved successfully! ")

    origin_circ = circuit
    all_measures = []
    for gate in circuit:
        if isinstance(gate, Measure):
            all_measures.append(gate)
    if circuit.has_measure_gate:
        circuit = circuit.remove_measure()

    U = circuit.matrix()
    kraus = np.array([U])
    return origin_circ, kraus


def matrix_of_op(op, on_which_qubit, qubit_num):
    if on_which_qubit == 0:
        M = op
    else:
        M = I_matrix
    for i in range(1, qubit_num):
        if i == on_which_qubit:
            M = np.kron(M, op)
        else:
            M = np.kron(M, I_matrix)
    return M


def random_insert_ops(origin_circuit, nums_and_ops, with_ctrl=True, after_measure=False, shots=1):
    """Insert single-qubit operators at random points in the circuit.

    args:
        circuit: The quantum circuit in which the operators are to be inserted.
        nums_and_ops: [num_0, op_0], [num_1, op_1], ...
            Where num_i is the number of insertion operators op_i. num_i: int, op_i: gate.
        with_ctrl: Whether to allow operator insertion on control qubits.
        after_measure: Whether to allow insertion of an operator after a measurement gate.
        shots: The number of new circuits generated.

    returns:
        An iterator of the new circuit generated after inserting the operators.
    """
    print(nums_and_ops)

    origin_circuit = origin_circuit.remove_barrier()  # 去掉栅栏门
    available_indexes = []
    if after_measure:
        available_indexes = range(len(origin_circuit))
    else:
        for i, gate in enumerate(origin_circuit):
            if not isinstance(gate, Measure):
                available_indexes.append(i)

    final_circuit = []
    for _ in range(shots):
        nums, ops = [], []
        for i in range(len(nums_and_ops)):
            if len(nums_and_ops[i]) != 2:
                raise ValueError(
                    f'The format of the argment "nums_and_ops" should be "[num_0, op_0], [num_1, op_1], ....".')
            if nums_and_ops[i][0] > len(available_indexes):
                raise ValueError(
                    f'The number of positions allowed to insert channel should be less than {len(available_indexes)}, but get {nums_and_ops[i][0]}.')
            nums.append(nums_and_ops[i][0])
            ops.append(nums_and_ops[i][1])
        indexes = []
        for num in nums:
            tem = sorted(np.random.choice(available_indexes, size=num, replace=False))
            indexes.append(tem)

        random_circit = Circuit()
        qubits_num = origin_circuit.n_qubits
        kraus_ = []  # U
        selected_qubits = []
        for i, gate in enumerate(origin_circuit):
            if not isinstance(gate, Measure) or after_measure:
                random_circit += gate
                if i == 0:
                    circ_ = Circuit(gates=I.on(qubits_num - 1))  # for first
                    kraus_.append(circ_.matrix())
                elif isinstance(gate, BasicGate):
                    # gate = BasicGate(gate)  # isinstance(gate, BasicGate) == Ture
                    circ_ = Circuit(gates=[gate, I.on(qubits_num - 1)])  # for first
                    U_ = circ_.matrix()
                    for u in range(len(kraus_)):
                        kraus_[u] = U_ @ kraus_[u]
                # print("iter {}: {}".format(i, np.array(kraus_).shape))
                for j, tem_indexs in enumerate(indexes):
                    for k in tem_indexs:
                        if k == i:
                            qubits = (gate.ctrl_qubits + gate.obj_qubits) if with_ctrl else gate.obj_qubits
                            qubit = int(np.random.choice(qubits))
                            if qubit in selected_qubits:  # the `qubit` has been selected.
                                continue
                            noise_op = ops[j]
                            random_circit += noise_op.on(qubit)
                            # print("the kraus of {}: ".format(noise_op))
                            # print(noise_op.matrix())
                            new_kraus = []
                            for u in kraus_:
                                for e in noise_op.matrix():
                                    # e_ = matrix_of_op(e, qubit, qubits_num)
                                    e_ = np.kron(np.eye(2 ** qubit), np.kron(e, np.eye(2 ** (qubits_num - qubit - 1))))
                                    new_kraus.append(e_ @ u)
                            kraus_ = new_kraus
                            selected_qubits.append(qubit)
                            print('len(selected_qubits) =', len(selected_qubits))
                            print('kraus_.shape =', np.array(kraus_).shape)
        final_circuit.append(random_circit)
        print(np.array(kraus_).shape)
    return Circuit(final_circuit[0]), np.array(kraus_)


def generating_circuit_with_random_noise(circ, model_name_):
    # generate random noise
    # noise_num = random.randint(1, len(circ))
    noise_num = circ.n_qubits
    print('add {} noise'.format(noise_num))
    ops = []
    left_noise_num = noise_num
    while left_noise_num > 0:
        noise = noise_op_map[random.choice(noise_ops)]
        # print(noise.__name__)
        p = float(round(random.uniform(0, 0.2), 5))
        noise_op = noise(p)
        temp_noise_num = random.randint(0, left_noise_num)
        # print('temp_noise_num =', temp_noise_num)
        if temp_noise_num != 0:
            ops.append([temp_noise_num, noise_op])
            left_noise_num -= temp_noise_num
            # print('[{}, {}]'.format(temp_noise_num, noise_op))
            # print('left_noise_num =', left_noise_num)

    all_measures = []
    for gate in circ:
        if isinstance(gate, Measure):
            all_measures.append(gate)
    if circ.has_measure_gate:
        circ = circ.remove_measure()

    # insert random noise
    circ, kraus_ = random_insert_ops(circ, ops)
    file_name_ = '{}_random.svg'.format(model_name_)

    for m in all_measures:
        circ += m

    # circ.svg().to_file("./figures/" + file_name_)  # qasm_file chop '.qasm'
    # print(file_name_ + " saved successfully! ")

    return circ, kraus_


def generating_circuit_with_specified_noise(circ, origin_kraus_, noise, noise_list_, kraus_file_, noise_p_: float,
                                            model_name_):
    all_measures = []
    for gate in circ:
        if isinstance(gate, Measure):
            all_measures.append(gate)
    circ = circ.remove_measure()
    # U = circ.matrix()
    qubits = range(circ.n_qubits)
    print("The noise type is:", noise)
    if noise == "mixed":
        # get all kraus operators
        E = noise_op_map[noise_list_[0]](noise_p_).matrix()
        kraus_ = E
        l = len(noise_list_)
        for q in qubits[::l]:
            for i in range(l):
                if q + i >= circ.n_qubits:
                    break
                noise_op = noise_op_map[noise_list_[i]]
                circ += noise_op(noise_p_).on(q + i)
                if q == 0 and i == 0:
                    continue
                new_kraus = []
                for m in kraus_:
                    for e in noise_op(noise_p_).matrix():
                        new_kraus.append(np.kron(m, e))
                kraus_ = new_kraus
        noise_list_ = [noise_op_map[i].__name__ for i in noise_list_]
        noise_list_ = [i[0: i.index("Channel")] for i in noise_list_]
        noise_name_ = "mixed_{}".format('_'.join(noise_list_))
    elif noise == "custom":
        data = load(kraus_file_)
        kraus_ = data['kraus']
        for i in range(kraus_.shape[0]):
            if kraus_[i].shape[0] != circ.n_qubits or kraus_[i].shape[1] != circ.n_qubits:
                raise RuntimeError("The dimension of the kraus operator is {}, not consistent with "
                                   "the circuit's ({}, {})! ".format(kraus_[i].shape, 2 ** circ.n_qubits,
                                                                     2 ** circ.n_qubits))
        noise_name_ = "custom_{}".format(kraus_file_[kraus_file_.rfind('/') + 1:-4])
    else:
        noise_op = noise_op_map[noise]
        noise_ = noise_op.__name__
        noise_ = noise_[0: noise_.index("Channel")]
        E = noise_op(noise_p_).matrix()
        kraus_ = E
        if noise_ == "Depolarizing":
            for q in qubits[::2]:
                circ += noise_op(noise_p_).on(q)
                new_kraus = []
                if q != 0:
                    # print(q + 1)
                    # print(len(kraus))
                    for i in kraus_:
                        for e in E:
                            new_kraus.append(np.kron(i, e))
                    kraus_ = new_kraus
                new_kraus = []
                for i in kraus_:
                    new_kraus.append(np.kron(i, I_matrix))
                kraus_ = new_kraus

        else:
            for q in qubits:
                circ += noise_op(noise_p_).on(q)
                # get all kraus operators
                if q != 0:
                    new_kraus = []
                    for i in kraus_:
                        for e in E:
                            new_kraus.append(np.kron(i, e))
                    kraus_ = new_kraus
        noise_name_ = noise_

    # print(len(kraus))
    # print(kraus[0].shape)
    # for i in range(len(kraus_)):
    #     kraus_[i] = kraus_[i] @ U
    new_kraus_ = []
    for i in range(len(origin_kraus_)):
        for j in range(len(kraus_)):
            # origin_kraus_[i] = kraus_[j] @ origin_kraus_[i]
            new_kraus_.append(kraus_[j] @ origin_kraus_[i])

    print("add {} with probability {}".format(noise, noise_p_))
    for m in all_measures:
        circ += m

    # file_name_ = '{}_{}_{}.svg'.format(model_name_, noise_name_, noise_p_)
    # circ.svg().to_file("./figures/" + file_name_)  # qasm_file chop '.qasm'
    # print(file_name_ + " saved successfully! ")

    new_kraus_ = np.array(new_kraus_)
    print(new_kraus_.shape)
    return new_kraus_, noise_name_


def evaluate():
    iter_num = 2
    GET_NEW_DATASET = False

    if '.npz' in str(argv[1]):
        # for example:
        # python3 batch_check.py binary_cav.npz 0.001 1 mixed
        data_file = str(argv[1])
        eps = float(argv[2])
        # n = int(argv[3])
        state_flag = str(argv[4])

        DATA = load(data_file)
        kraus = DATA['kraus']
        O = DATA['O']
        data = DATA['data']
        label = DATA['label']
        type = 'npz'
        # origin_dataset_size = label.shape[0]
        model_name = data_file[data_file.rfind('/') + 1: data_file.rfind('_')]
        # file_name = '{}_{}_{}_{}.csv'.format(model_name, eps, n, state_flag)  # 默认文件名

        verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

        with open("./results/adversary_training.csv", "a+") as csvfile:
            w = csv.writer(csvfile)
            c_eps = eps
            for i in range(iter_num):
                origin_ac_temp, origin_time_temp = verifier(kraus, O, data, label, c_eps, type)
                # verifier(kraus, O, data, label, c_eps, type, GET_NEW_DATASET, origin_dataset_size)
                origin_ac_1 = origin_ac_temp[0] * 100
                origin_ac_2 = origin_ac_temp[1] * 100
                origin_time_1 = origin_time_temp[0]
                origin_time_2 = origin_time_temp[1]
                w.writerow([model_name, c_eps, 'c_0',
                            '%.2f' % origin_ac_1, '%.4f' % origin_time_1,
                            '%.2f' % origin_ac_2, '%.4f' % origin_time_2])
                gc.collect()

            for i in range(iter_num):
                random_ac_temp, random_time_temp = verifier(kraus, O, data, label, c_eps, type)
                random_ac_1 = random_ac_temp[0] * 100
                random_ac_2 = random_ac_temp[1] * 100
                random_time_1 = random_time_temp[0]
                random_time_2 = random_time_temp[1]
                w.writerow([model_name, c_eps, 'c_1',
                            '%.2f' % random_ac_1, '%.4f' % random_time_1,
                            '%.2f' % random_ac_2, '%.4f' % random_time_2])
                gc.collect()

            noise_ = 'depolarizing'
            noise_probs = [0.001, 0.005]
            for noise_p in noise_probs:
                for i in range(iter_num):
                    final_ac_temp, final_time_temp = verifier(kraus, O, data, label, c_eps, type)
                    final_ac_1 = final_ac_temp[0] * 100
                    final_ac_2 = final_ac_temp[1] * 100
                    final_time_1 = final_time_temp[0]
                    final_time_2 = final_time_temp[1]
                    w.writerow([model_name, c_eps, 'c_2 ({}_{})'.format(noise_, noise_p),
                                '%.2f' % final_ac_1, '%.4f' % final_time_1,
                                '%.2f' % final_ac_2, '%.4f' % final_time_2])
                    gc.collect()
    else:
        # '.qasm' in str(argv[1])
        qasm_file = str(argv[1])
        data_file = str(argv[2])
        state_flag = str(argv[3])
        # noise_type = argv[5]
        # n = 1

        DATA = load(data_file)
        O = DATA['O']
        data = DATA['data']
        label = DATA['label']
        type = 'qasm'

        model_name = qasm_file[qasm_file.rfind('/') + 1:-5]
        verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

        digits = '36'
        ADVERSARY_EXAMPLE = False
        TEST_MNIST = ('mnist' in data_file)
        if TEST_MNIST:  # digits != '36'
            # ADVERSARY_EXAMPLE = (str(argv[4]) == 'true')
            if '_data' in data_file:  # digits != '36'
                digits = data_file[data_file.rfind('_data') - 2: data_file.rfind('_data')]

        origin_circuit, origin_kraus = qasm2mq(qasm_file)
        if model_name != "fashion10":
            random_circuit, random_kraus = generating_circuit_with_random_noise(origin_circuit, model_name)

        epss = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]
        probs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
        noise_types = ["bit_flip", "depolarizing", "phase_flip", "mixed"]
        kraus_file = None
        with open("./results/local_results.csv", "a+") as csvfile:
            w = csv.writer(csvfile)
            for c_eps in epss:
                if TEST_MNIST:
                    origin_ac_temp, origin_time_temp = verifier(origin_kraus, O, data, label, c_eps, type,
                                                                GET_NEW_DATASET, 0, ADVERSARY_EXAMPLE,
                                                                digits, 'mnist')
                else:
                    origin_ac_temp, origin_time_temp = verifier(origin_kraus, O, data, label, c_eps, type,
                                                                GET_NEW_DATASET, 0)
                origin_ac_1 = origin_ac_temp[0] * 100
                origin_ac_2 = origin_ac_temp[1] * 100
                origin_time_1 = origin_time_temp[0]
                origin_time_2 = origin_time_temp[1]
                w.writerow([model_name, 'c_0', c_eps,
                            '%.2f' % origin_ac_1, '%.4f' % origin_time_1,
                            '%.2f' % origin_ac_2, '%.4f' % origin_time_2])
            for c_eps in epss:
                if TEST_MNIST:
                    random_ac_temp, random_time_temp = verifier(random_kraus, O, data, label, c_eps, type,
                                                                GET_NEW_DATASET, 0, ADVERSARY_EXAMPLE,
                                                                digits, 'mnist')
                else:
                    random_ac_temp, random_time_temp = verifier(random_kraus, O, data, label, c_eps, type,
                                                                GET_NEW_DATASET, 0)
                random_ac_1 = random_ac_temp[0] * 100
                random_ac_2 = random_ac_temp[1] * 100
                random_time_1 = random_time_temp[0]
                random_time_2 = random_time_temp[1]
                w.writerow([model_name, 'c_1', c_eps,
                            '%.2f' % random_ac_1, '%.4f' % random_time_1,
                            '%.2f' % random_ac_2, '%.4f' % random_time_2])

            # probs_iris = [0.01, 0.01, 0.05, 0.05]
            # epss_iris = [0.005, 0.05, 0.003, 0.01]
            # probs_fashion8 = [0.01, 0.01, 0.05, 0.05]
            # epss_fashion8 = [0.001, 0.01, 0.001, 0.005]
            # probs_fashion10 = [0.001]
            # epss_fashion10 = [0.0001]
            # probs_mnist13 = [0.001, 0.001, 0.01, 0.01]
            # epss_mnist13 = [0.001, 0.003, 0.0001, 0.001]
            # probs_tfi4 = [0.01, 0.01, 0.05, 0.05]
            # epss_tfi4 = [0.001, 0.005, 0.005, 0.01]
            # probs_tfi8 = [0.01, 0.01, 0.05, 0.05]
            # epss_tfi8 = [0.001, 0.01, 0.005, 0.01]
            # probs_tfi12 = [0.1, 0.15, 0.03, 0.175]
            # epss_tfi12 = [0.003, 0.005, 0.01, 0.0001]
            # prob_map = {
            #     'iris': probs_iris,
            #     'fashion8': probs_fashion8,
            #     'fashion10': probs_fashion10,
            #     'mnist13': probs_mnist13,
            #     'tfi4': probs_tfi4,
            #     'tfi8': probs_tfi8,
            #     'tfi12': probs_tfi12,
            # }
            # eps_map = {
            #     'iris': epss_iris,
            #     'fashion8': epss_fashion8,
            #     'fashion10': epss_fashion10,
            #     'mnist13': epss_mnist13,
            #     'tfi4': epss_tfi4,
            #     'tfi8': epss_tfi8,
            #     'tfi12': epss_tfi12,
            # }
            # probs = prob_map[model_name]
            # epss = eps_map[model_name]

            # for noise_p, c_eps in zip(probs, epss):
            for noise_type in noise_types:
                noise_list = ["bit_flip", "depolarizing", "phase_flip"] if noise_type == 'mixed' else []
                for noise_p in probs:
                    if model_name == "fashion10":
                        # np.savez('./fashion10_random_kraus.npz', random_kraus=random_kraus)
                        # np.savez('./fashion10_final_kraus.npz', final_kraus=final_kraus)
                        random_circuit = None  # no need
                        final_circuit = None  # no need
                        random_kraus = load('./fashion10_random_kraus.npz')['random_kraus']
                        final_kraus = load('./fashion10_final_kraus.npz')['final_kraus']
                    else:
                        final_kraus, noise_name = generating_circuit_with_specified_noise(random_circuit, random_kraus,
                                                                                          noise_type, noise_list,
                                                                                          kraus_file, noise_p,
                                                                                          model_name)
                        noise_ = noise_type.replace('_', '-')
                        for c_eps in epss:
                            if TEST_MNIST:
                                final_ac_temp, final_time_temp = verifier(final_kraus, O, data, label, c_eps, type,
                                                                          GET_NEW_DATASET, 0, ADVERSARY_EXAMPLE,
                                                                          digits, 'mnist')
                            else:
                                final_ac_temp, final_time_temp = verifier(final_kraus, O, data, label, c_eps, type,
                                                                          GET_NEW_DATASET, 0)
                            final_ac_1 = final_ac_temp[0] * 100
                            final_ac_2 = final_ac_temp[1] * 100
                            final_time_1 = final_time_temp[0]
                            final_time_2 = final_time_temp[1]
                            w.writerow([model_name, 'c_2 ({}_{})'.format(noise_, noise_p), c_eps,
                                        '%.2f' % final_ac_1, '%.4f' % final_time_1,
                                        '%.2f' % final_ac_2, '%.4f' % final_time_2])
                            gc.collect()


def evaluate_all_mnist():
    state_flag = 'pure'
    verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier
    ADVERSARY_EXAMPLE = False
    GET_NEW_DATASET = False
    noise_types = ["bit_flip", "depolarizing", "phase_flip", "mixed"]
    epss = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]
    probs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
    noise_types = ["bit_flip", "depolarizing", "phase_flip", "mixed"]
    for d0 in range(0, 10):
        for d1 in range(d0 + 1, 10):
            digits = str(d0) + str(d1)
            if digits == '13':
                continue

            data_file = './model_and_data/mnist{}_data.npz'.format(digits)
            model_name = data_file[data_file.rfind('/') + 1: data_file.rfind('_')]
            DATA = load(data_file)
            O = DATA['O']
            data = DATA['data']
            label = DATA['label']
            type = 'qasm'

            c_eps = random.choice(epss)
            kraus_file = None

            qasm_file = './model_and_data/' + model_name + '.qasm'
            origin_circuit, origin_kraus = qasm2mq(qasm_file)
            random_circuit, random_kraus = generating_circuit_with_random_noise(origin_circuit, model_name)

            origin_ac_temp, origin_time_temp = verifier(origin_kraus, O, data, label, c_eps, type,
                                                        GET_NEW_DATASET, 0, ADVERSARY_EXAMPLE,
                                                        digits, 'mnist')
            random_ac_temp, random_time_temp = verifier(random_kraus, O, data, label, c_eps, type,
                                                        GET_NEW_DATASET, 0, ADVERSARY_EXAMPLE,
                                                        digits, 'mnist')
            origin_ac_1 = origin_ac_temp[0] * 100
            origin_ac_2 = origin_ac_temp[1] * 100
            origin_time_1 = origin_time_temp[0]
            origin_time_2 = origin_time_temp[1]
            random_ac_1 = random_ac_temp[0] * 100
            random_ac_2 = random_ac_temp[1] * 100
            random_time_1 = random_time_temp[0]
            random_time_2 = random_time_temp[1]
            with open("./results/local_results.csv", "a+") as csvfile:
                w = csv.writer(csvfile)
                w.writerows([
                    [model_name, 'c_0', c_eps,
                     '%.2f' % origin_ac_1, '%.4f' % origin_time_1,
                     '%.2f' % origin_ac_2, '%.4f' % origin_time_2],
                    [model_name, 'c_1', c_eps,
                     '%.2f' % random_ac_1, '%.4f' % random_time_1,
                     '%.2f' % random_ac_2, '%.4f' % random_time_2],
                ])
            for _ in range(1):
                noise_type = random.choice(noise_types)
                noise_list = ["bit_flip", "depolarizing", "phase_flip"] if noise_type == 'mixed' else []
                # noise_p = random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075])
                # eps = random.choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01])
                # eps = choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075])
                noise_p = random.choice(probs)
                print('*' * 40 + "verifying {} with {}_{}".format(model_name, noise_type, noise_p) + '*' * 40)
                final_kraus, noise_name = generating_circuit_with_specified_noise(random_circuit, random_kraus,
                                                                                  noise_type, noise_list,
                                                                                  kraus_file, noise_p, model_name)
                final_ac_temp, final_time_temp = verifier(final_kraus, O, data, label, c_eps, type,
                                                          GET_NEW_DATASET, 0, ADVERSARY_EXAMPLE,
                                                          digits, 'mnist')
                final_ac_1 = final_ac_temp[0] * 100
                final_ac_2 = final_ac_temp[1] * 100
                final_time_1 = final_time_temp[0]
                final_time_2 = final_time_temp[1]
                noise_ = noise_type.replace('_', '-')
                with open("./results/local_results.csv", "a+") as csvfile:
                    w = csv.writer(csvfile)
                    w.writerow([model_name, 'c_2 ({}_{})'.format(noise_, noise_p), c_eps,
                                '%.2f' % final_ac_1, '%.4f' % final_time_1,
                                '%.2f' % final_ac_2, '%.4f' % final_time_2])
                gc.collect()


def adversary_training_evaluate_mnist():
    n = 1
    state_flag = 'pure'
    ADVERSARY_EXAMPLE = True

    verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

    for d0 in range(0, 10):
        for d1 in range(d0 + 1, 10):
            digits = str(d0) + str(d1)
            if digits == '13':
                continue

            data_file = './model_and_data/mnist{}_data.npz'.format(digits)
            # data_file = './model_and_data/TFIchain8_data.npz'
            model_name = data_file[data_file.rfind('/') + 1: data_file.rfind('_')]
            DATA = load(data_file)
            O = DATA['O']
            data = DATA['data']
            label = DATA['label']

            noise_type = random.choice(["bit_flip", "depolarizing", "phase_flip", "mixed"])
            iter_num = 2

            noise_p = random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075])
            eps = random.choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01])
            type = 'qasm'
            # eps = choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075])

            print('noise_type =', noise_type)
            print('noise_p =', noise_p)
            print('eps =', eps)

            noise_list = []
            kraus_file = None
            if noise_type == 'mixed':
                noise_list = ["bit_flip", "depolarizing", "phase_flip"]

            qasm_file = './model_and_data/' + model_name + '.qasm'
            origin_circuit, origin_kraus = qasm2mq(qasm_file)
            random_circuit, random_kraus = generating_circuit_with_random_noise(origin_circuit, model_name)
            final_kraus, noise_name = generating_circuit_with_specified_noise(random_circuit, random_kraus, noise_type,
                                                                              noise_list,
                                                                              kraus_file, noise_p, model_name)

            noise_ = noise_type.replace('_', '-')
            with open("./results/adversary_training.csv", "a+") as csvfile:
                w = csv.writer(csvfile)
                c_eps = eps
                for i in range(iter_num):
                    print('kraus.shape =', final_kraus.shape)
                    print('data.shape =', data.shape)
                    print('label.shape =', label.shape)
                    print('O.shape =', O.shape)
                    # origin_ac_temp, origin_time_temp, new_labels = verifier(origin_kraus, O, data, label, c_eps, type,
                    #                                                         ADVERSARY_EXAMPLE, digits, 'mnist')
                    # random_ac_temp, random_time_temp, new_labels = verifier(random_kraus, O, data, label, c_eps, type,
                    #                                                         ADVERSARY_EXAMPLE, digits, 'mnist')
                    final_ac_temp, final_time_temp, new_data, new_labels = verifier(final_kraus, O, data, label, c_eps,
                                                                                    type,
                                                                                    ADVERSARY_EXAMPLE, digits, 'mnist')

                    data = new_data
                    label = new_labels

                    # origin_ac_1 = origin_ac_temp[0] * 100
                    # origin_ac_2 = origin_ac_temp[1] * 100
                    # origin_time_1 = origin_time_temp[0]
                    # origin_time_2 = origin_time_temp[1]
                    #
                    # random_ac_1 = random_ac_temp[0] * 100
                    # random_ac_2 = random_ac_temp[1] * 100
                    # random_time_1 = random_time_temp[0]
                    # random_time_2 = random_time_temp[1]

                    final_ac_1 = final_ac_temp[0] * 100
                    final_ac_2 = final_ac_temp[1] * 100
                    final_time_1 = final_time_temp[0]
                    final_time_2 = final_time_temp[1]
                    w.writerow([model_name, '{}_{}_{}'.format(noise_, noise_p, c_eps),
                                'i_{}'.format(i),
                                '%.2f' % final_ac_1, '%.4f' % final_time_1,
                                '%.2f' % final_ac_2, '%.4f' % final_time_2])


def adversary_training_evaluate():
    iter_num = 2
    GET_NEW_DATASET = True

    if '.npz' in str(argv[1]):
        # for example:
        # python evaluate.py ./model_and_data/qubit_cav.npz 0.001 1 mixed
        data_file = str(argv[1])
        eps = float(argv[2])
        n = int(argv[3])
        state_flag = str(argv[4])
        c_eps = eps

        DATA = load(data_file)
        kraus = DATA['kraus']
        O = DATA['O']
        data = DATA['data']
        label = DATA['label']
        type = 'npz'
        origin_dataset_size = label.shape[0]
        model_name = data_file[data_file.rfind('/') + 1: data_file.rfind('_')]
        file_name = '{}_{}_{}_{}.csv'.format(model_name, eps, n, state_flag)  # 默认文件名

        verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

        with open("./results/adversary_training.csv", "a+") as csvfile:
            w = csv.writer(csvfile)
            for i in range(iter_num):
                final_ac_temp, final_time_temp, new_data, new_labels, non_robust_num_1, non_robust_num_2 = \
                    verifier(kraus, O, data, label, c_eps, type, GET_NEW_DATASET, origin_dataset_size)
                data = copy.deepcopy(new_data)
                label = copy.deepcopy(new_labels)
                final_ac_1 = final_ac_temp[0] * 100
                final_ac_2 = final_ac_temp[1] * 100
                final_time_1 = final_time_temp[0]
                final_time_2 = final_time_temp[1]
                # noise_ = noise_type.replace('_', '-')
                w.writerow([model_name, c_eps, 'c_0', 'V_{}'.format(i),
                            non_robust_num_1, '%.2f' % final_ac_1, '%.4f' % final_time_1,
                            non_robust_num_2, '%.2f' % final_ac_2, '%.4f' % final_time_2])

            data = DATA['data']
            label = DATA['label']
            for i in range(iter_num):
                final_ac_temp, final_time_temp, new_data, new_labels, non_robust_num_1, non_robust_num_2 = \
                    verifier(kraus, O, data, label, c_eps, type, GET_NEW_DATASET, origin_dataset_size)
                data = copy.deepcopy(new_data)
                label = copy.deepcopy(new_labels)
                final_ac_1 = final_ac_temp[0] * 100
                final_ac_2 = final_ac_temp[1] * 100
                final_time_1 = final_time_temp[0]
                final_time_2 = final_time_temp[1]
                w.writerow([model_name, c_eps, 'c_1', 'V_{}'.format(i),
                            non_robust_num_1, '%.2f' % final_ac_1, '%.4f' % final_time_1,
                            non_robust_num_2, '%.2f' % final_ac_2, '%.4f' % final_time_2])

            data = DATA['data']
            label = DATA['label']
            noise_ = 'depolarizing'
            noise_probs = [0.001, 0.005]
            for noise_p in noise_probs:
                for i in range(iter_num):
                    final_ac_temp, final_time_temp, new_data, new_labels, non_robust_num_1, non_robust_num_2 = \
                        verifier(kraus, O, data, label, c_eps, type, GET_NEW_DATASET, origin_dataset_size)
                    data = copy.deepcopy(new_data)
                    label = copy.deepcopy(new_labels)
                    final_ac_1 = final_ac_temp[0] * 100
                    final_ac_2 = final_ac_temp[1] * 100
                    final_time_1 = final_time_temp[0]
                    final_time_2 = final_time_temp[1]
                    w.writerow([model_name, c_eps, 'c_2 ({}_{})'.format(noise_, noise_p), 'V_{}'.format(i),
                                non_robust_num_1, '%.2f' % final_ac_1, '%.4f' % final_time_1,
                                non_robust_num_2, '%.2f' % final_ac_2, '%.4f' % final_time_2])

    else:
        # '.qasm' in str(argv[1])
        qasm_file = str(argv[1])
        data_file = str(argv[2])
        state_flag = str(argv[3])
        noise_type = argv[5]

        model_name = qasm_file[qasm_file.rfind('/') + 1:-5]
        verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

        digits = '36'
        ADVERSARY_EXAMPLE = False
        if 'mnist' in data_file:  # digits != '36'
            ADVERSARY_EXAMPLE = (str(argv[4]) == 'true')
            if '_data' in data_file:  # digits != '36'
                digits = data_file[data_file.rfind('_data') - 2: data_file.rfind('_data')]

        noise_list = []
        kraus_file = None
        if noise_type == 'mixed':
            noise_list = ["bit_flip", "depolarizing", "phase_flip"]
        # arg_num = len(argv)
        # if noise_type == 'mixed':
        #     noise_list = [i for i in argv[8: arg_num - 1]]
        #     print("noise_list: ", noise_list)
        # elif noise_type == 'custom':
        #     kraus_file = argv[8]
        origin_circuit, origin_kraus = qasm2mq(qasm_file)
        # if model_name not in ['mnist13', 'tfi4', 'tfi8', 'fashion10']:
        # if model_name not in ['fashion10']:
        random_circuit, random_kraus = generating_circuit_with_random_noise(origin_circuit, model_name)

        # probs_iris = [0.01, 0.01, 0.05, 0.05]
        # epss_iris = [0.005, 0.05, 0.003, 0.01]
        # probs_fashion8 = [0.01, 0.01, 0.05, 0.05]
        # epss_fashion8 = [0.001, 0.01, 0.001, 0.005]
        # probs_fashion10 = [0.001]
        # epss_fashion10 = [0.0001]
        # probs_mnist13 = [0.001, 0.001, 0.01, 0.01]
        # epss_mnist13 = [0.001, 0.003, 0.0001, 0.001]
        # probs_tfi4 = [0.01, 0.01, 0.05, 0.05]
        # epss_tfi4 = [0.001, 0.005, 0.005, 0.01]
        # probs_tfi8 = [0.01, 0.01, 0.05, 0.05]
        # epss_tfi8 = [0.001, 0.01, 0.005, 0.01]
        # probs_tfi12 = [0.1, 0.15, 0.03, 0.175]
        # epss_tfi12 = [0.003, 0.005, 0.01, 0.0001]

        probs_iris = [0.01, 0.05]
        epss_iris = 0.005
        probs_fashion8 = [0.01, 0.05]
        epss_fashion8 = 0.001
        probs_fashion10 = [0.001, 0.005]
        epss_fashion10 = 0.0001
        probs_mnist13 = [0.001, 0.01]
        epss_mnist13 = 0.003
        probs_tfi4 = [0.01, 0.05]
        epss_tfi4 = 0.005
        probs_tfi8 = [0.01, 0.05]
        epss_tfi8 = 0.001

        prob_map = {
            'iris': probs_iris,
            'fashion8': probs_fashion8,
            'fashion10': probs_fashion10,
            'mnist13': probs_mnist13,
            'tfi4': probs_tfi4,
            'tfi8': probs_tfi8,
            # 'tfi12': probs_tfi12,
        }
        eps_map = {
            'iris': epss_iris,
            'fashion8': epss_fashion8,
            'fashion10': epss_fashion10,
            'mnist13': epss_mnist13,
            'tfi4': epss_tfi4,
            'tfi8': epss_tfi8,
            # 'tfi12': epss_tfi12,
        }
        probs = prob_map[model_name]
        c_eps = eps_map[model_name]
        prob_1 = probs[0]
        prob_2 = probs[1]
        if model_name == "fashion10":
            np.savez('./fashion10_random_kraus.npz', random_kraus=random_kraus)
            # np.savez('./fashion10_final_kraus.npz', final_kraus=final_kraus)
            # random_kraus = load('./fashion10_random_kraus.npz')['random_kraus']
            # random_circuit = origin_circuit
            # final_kraus = load('./fashion10_final_kraus.npz')['final_kraus']
        # elif model_name not in ['mnist13', 'tfi4', 'tfi8']:
        final_kraus_1, noise_name = generating_circuit_with_specified_noise(random_circuit, random_kraus,
                                                                            noise_type, noise_list,
                                                                            kraus_file, prob_1, model_name)
        final_kraus_2, noise_name = generating_circuit_with_specified_noise(random_circuit, random_kraus,
                                                                            noise_type, noise_list,
                                                                            kraus_file, prob_2, model_name)
        DATA = load(data_file)
        O = DATA['O']
        data = DATA['data']
        label = DATA['label']
        type = 'qasm'

        origin_dataset_size = label.shape[0]
        noise_ = noise_type.replace('_', '-')
        # data_c0, data_c1, data_c2_1, data_c2_2 = data, data, data, data
        # label_c0, label_c1, label_c2_1, label_c2_2 = label, label, label, label
        data_c0, label_c0 = copy.deepcopy(data), copy.deepcopy(label)
        data_c1, label_c1 = copy.deepcopy(data), copy.deepcopy(label)
        data_c2_1, label_c2_1 = copy.deepcopy(data), copy.deepcopy(label)
        data_c2_2, label_c2_2 = copy.deepcopy(data), copy.deepcopy(label)
        with (open("./results/adversary_training.csv", "a+") as csvfile):
            w = csv.writer(csvfile)
            for i in range(iter_num):
                if 'mnist' in model_name:
                    origin_ac_temp, origin_time_temp, new_data_c0, new_labels_c0, non_robust_num_1_c0, non_robust_num_2_c0 = \
                        verifier(origin_kraus, O, data_c0, label_c0, c_eps, type,
                                 GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
                    random_ac_temp, random_time_temp, new_data_c1, new_labels_c1, non_robust_num_1_c1, non_robust_num_2_c1 = \
                        verifier(random_kraus, O, data_c1, label_c1, c_eps, type,
                                 GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
                    final_ac_temp_1, final_time_temp_1, new_data_c2_1, new_labels_c2_1, non_robust_num_1_c21, non_robust_num_2_c21 = \
                        verifier(final_kraus_1, O, data_c2_1, label_c2_1, c_eps, type,
                                 GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
                    final_ac_temp_2, final_time_temp_2, new_data_c2_2, new_labels_c2_2, non_robust_num_1_c22, non_robust_num_2_c22 = \
                        verifier(final_kraus_2, O, data_c2_2, label_c2_2, c_eps, type,
                                 GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
                # elif model_name in ['tfi4', 'tfi8']:
                #     final_ac_temp, final_time_temp, new_data_c0, new_labels_c0 = verifier(origin_kraus, O, data_c0,
                #                                                                           label_c0,
                #                                                                           c_eps, type)
                else:
                    origin_ac_temp, origin_time_temp, new_data_c0, new_labels_c0, non_robust_num_1_c0, non_robust_num_2_c0 = \
                        verifier(origin_kraus, O, data_c0, label_c0, c_eps, type, GET_NEW_DATASET, origin_dataset_size)
                    random_ac_temp, random_time_temp, new_data_c1, new_labels_c1, non_robust_num_1_c1, non_robust_num_2_c1 = \
                        verifier(random_kraus, O, data_c1, label_c1, c_eps, type, GET_NEW_DATASET, origin_dataset_size)
                    final_ac_temp_1, final_time_temp_1, new_data_c2_1, new_labels_c2_1, non_robust_num_1_c21, non_robust_num_2_c21 = \
                        verifier(final_kraus_1, O, data_c2_1, label_c2_1, c_eps, type, GET_NEW_DATASET,
                                 origin_dataset_size)
                    final_ac_temp_2, final_time_temp_2, new_data_c2_2, new_labels_c2_2, non_robust_num_1_c22, non_robust_num_2_c22 = \
                        verifier(final_kraus_2, O, data_c2_2, label_c2_2, c_eps, type, GET_NEW_DATASET,
                                 origin_dataset_size)

                data_c0, label_c0 = copy.deepcopy(new_data_c0), copy.deepcopy(new_labels_c0)
                data_c1, label_c1 = copy.deepcopy(new_data_c1), copy.deepcopy(new_labels_c1)
                data_c2_1, label_c2_1 = copy.deepcopy(new_data_c2_1), copy.deepcopy(new_labels_c2_1)
                data_c2_2, label_c2_2 = copy.deepcopy(new_data_c2_2), copy.deepcopy(new_labels_c2_2)
                np.savez('./model_and_data/iris_newdata_c0.npz', O=O, data=new_data_c0, label=new_labels_c0)
                np.savez('./model_and_data/iris_newdata_c1.npz', O=O, data=new_data_c1, label=new_labels_c1)
                np.savez('./model_and_data/iris_newdata_c21.npz', O=O, data=new_data_c2_1, label=new_labels_c2_1)
                np.savez('./model_and_data/iris_newdata_c22.npz', O=O, data=new_data_c2_2, label=new_labels_c2_2)

                origin_ac_1 = origin_ac_temp[0] * 100
                origin_ac_2 = origin_ac_temp[1] * 100
                origin_time_1 = origin_time_temp[0]
                origin_time_2 = origin_time_temp[1]

                random_ac_1 = random_ac_temp[0] * 100
                random_ac_2 = random_ac_temp[1] * 100
                random_time_1 = random_time_temp[0]
                random_time_2 = random_time_temp[1]

                final_ac_1_c21 = final_ac_temp_1[0] * 100
                final_ac_2_c21 = final_ac_temp_1[1] * 100
                final_time_1_c21 = final_time_temp_1[0]
                final_time_2_c21 = final_time_temp_1[1]

                final_ac_1_c22 = final_ac_temp_2[0] * 100
                final_ac_2_c22 = final_ac_temp_2[1] * 100
                final_time_1_c22 = final_time_temp_2[0]
                final_time_2_c22 = final_time_temp_2[1]

                # w.writerow([model_name, '{}_{}_{}'.format(noise_, noise_p, c_eps),
                #             'i_{}'.format(i),
                #             '%.2f' % final_ac_1, '%.4f' % final_time_1,
                #             '%.2f' % final_ac_2, '%.4f' % final_time_2])
                # w.writerows([
                #     [model_name, '{}_{}_{}'.format(noise_, noise_p, c_eps), 'c_0', 'i_{}'.format(i),
                #      '%.2f' % origin_ac_1, '%.4f' % origin_time_1, '%.2f' % origin_ac_2, '%.4f' % origin_time_2],
                #     [model_name, '{}_{}_{}'.format(noise_, noise_p, c_eps), 'c_1', 'i_{}'.format(i),
                #      '%.2f' % random_ac_1, '%.4f' % random_time_1, '%.2f' % random_ac_2, '%.4f' % random_time_2],
                #     [model_name, '{}_{}_{}'.format(noise_, noise_p, c_eps), 'c_2', 'i_{}'.format(i),
                #      '%.2f' % final_ac_1, '%.4f' % final_time_1, '%.2f' % final_ac_2, '%.4f' % final_time_2]
                # ])
                w.writerows([
                    [model_name, c_eps, 'c_0', 'i_{}'.format(i),
                     non_robust_num_1_c0, '%.2f' % origin_ac_1, '%.4f' % origin_time_1,
                     non_robust_num_2_c0, '%.2f' % origin_ac_2, '%.4f' % origin_time_2],
                    [model_name, c_eps, 'c_1', 'i_{}'.format(i),
                     non_robust_num_1_c1, '%.2f' % random_ac_1, '%.4f' % random_time_1,
                     non_robust_num_2_c1, '%.2f' % random_ac_2, '%.4f' % random_time_2],
                    [model_name, c_eps, 'c_2 ({}_{})'.format(noise_, prob_1), 'i_{}'.format(i),
                     non_robust_num_1_c21, '%.2f' % final_ac_1_c21, '%.4f' % final_time_1_c21,
                     non_robust_num_2_c21, '%.2f' % final_ac_2_c21, '%.4f' % final_time_2_c21],
                    [model_name, c_eps, 'c_2 ({}_{})'.format(noise_, prob_2), 'i_{}'.format(i),
                     non_robust_num_1_c22, '%.2f' % final_ac_1_c22, '%.4f' % final_time_1_c22,
                     non_robust_num_2_c22, '%.2f' % final_ac_2_c22, '%.4f' % final_time_2_c22]
                ])


if len(argv) > 1:
    evaluate()
else:
    evaluate_all_mnist()

# adversary_training_evaluate()
