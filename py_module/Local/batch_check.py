from VeriQ import RobustnessVerifier, PureRobustnessVerifier
from prettytable import PrettyTable
from sys import argv
import numpy as np
from numpy import load

from mindquantum.io import OpenQASM
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import BitFlipChannel, DepolarizingChannel, PhaseFlipChannel, Measure
from random import choice, uniform
import mindquantum
import gc
import csv
from multiprocessing import Pool

noise_op_mq = [
    PhaseFlipChannel,
    DepolarizingChannel,
    BitFlipChannel,
    DepolarizingChannel
]

noise_op_map = {
    "bit_flip": BitFlipChannel,
    "depolarizing": DepolarizingChannel,
    "phase_flip": PhaseFlipChannel,
    "mixed": DepolarizingChannel
}


def random_insert_ops(circuit, *nums_and_ops, with_ctrl=True, after_measure=False, shots=1):
    '''在线路随机位置插入单比特算符 ops。

    args:
        circuit: 待插入算符的量子线路;
        nums_and_ops: [num_0, op_0], [num_1, op_1], ...。其中,  num_i 为插入算符 op_i 的数量。num_i: int, op_i: gate;
        with_ctrl: 是否允许控制位上插入算符;
        after_measure: 是否允许在测量门后插入算符;
        shots: 生成新线路的数量。

    returns:
        可生成量子线路的迭代器。
    '''

    circuit = circuit.remove_barrier()  # 去掉栅栏门
    if after_measure:
        available_indexs = list(range(len(circuit)))
    else:
        available_indexs = []
        for i, gate in enumerate(circuit):
            if isinstance(gate, Measure):
                continue
            else:
                available_indexs.append(i)
    circs = []
    for _ in range(shots):
        nums, ops = [], []
        if isinstance(nums_and_ops[0], int):  # 只插入一种噪声门
            if nums_and_ops[0] > len(available_indexs):
                raise ValueError(
                    f'The number of positions allowed to insert channel should be less than {len(available_indexs)}, but get {nums_and_ops[0]}.')
            nums.append(nums_and_ops[0])
            ops.append(nums_and_ops[1])
        else:
            for i in range(len(nums_and_ops)):
                if len(nums_and_ops[i]) != 2:
                    raise ValueError(
                        f'The format of the argment "nums_and_ops" should be "[num_0, op_0], [num_1, op_1], ....".')
                if nums_and_ops[i][0] > len(available_indexs):
                    raise ValueError(
                        f'The number of positions allowed to insert channel should be less than {len(available_indexs)}, but get {nums_and_ops[i][0]}.')
                nums.append(nums_and_ops[i][0])
                ops.append(nums_and_ops[i][1])
        indexs = []
        for num in nums:
            tem = sorted(np.random.choice(available_indexs, size=num, replace=False))
            indexs.append(tem)

        circ = Circuit()
        for i, gate in enumerate(circuit):
            if isinstance(gate, Measure) and not after_measure:
                continue
            else:
                circ += gate
            for j, tem_indexs in enumerate(indexs):
                for k in tem_indexs:
                    if k == i:
                        if with_ctrl:
                            qubits = gate.ctrl_qubits + gate.obj_qubits
                        else:
                            qubits = gate.obj_qubits
                        qubit = np.random.choice(qubits)
                        circ += ops[j].on(int(qubit))
        circs.append(circ)
    return circs


def qasmstr2mq(qasm_str):
    circuit = OpenQASM().from_string(qasm_str)
    if circuit.parameterized:
        val_list = []
        for param in circuit.params_name:
            param = param.replace('pi', str(np.pi)).replace('π', str(np.pi))
            val_list.append(float(param))
        pr = dict(zip(circuit.params_name, val_list))  # 获取线路参数
        circuit = circuit.apply_value(pr)
    return circuit


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

    # model_name = "{}_model.svg".format(qasm_file[qasm_file.rfind('/') + 1:-5])
    # circuit.svg().to_file("./Figures/" + model_name)  # qasm_file chop '.qasm'
    # print(model_name + " saved successfully! ")

    if circuit.has_measure_gate:
        circuit = circuit.remove_measure()

    U = circuit.matrix()
    kraus = np.array([U])
    return kraus


I = mindquantum.core.gates.I.matrix()


def qasm2mq_with_specified_noise(file, noise, noise_list, kraus_file, p: float):
    f = open(file)
    qasm_str = f.read()
    f.close()
    circuit = OpenQASM().from_string(qasm_str)
    if circuit.parameterized:
        val_list = []
        for param in circuit.params_name:
            param = param.replace('pi', str(np.pi)).replace('π', str(np.pi))
            val_list.append(float(param))
        pr = dict(zip(circuit.params_name, val_list))  # 获取线路参数
        circuit = circuit.apply_value(pr)

    # print(circuit)
    all_measures = []
    for gate in circuit:
        # print(type(gate))
        if type(gate) is Measure:
            all_measures.append(gate)

    if circuit.has_measure_gate:
        circuit = circuit.remove_measure()
    U = circuit.matrix()

    # add random noise
    num = circuit.n_qubits
    n_qubits = range(num)
    print("The noise type is:", noise)
    if noise == "mixed":
        noise_ = noise
        # get all kraus operators
        E = noise_op_map[noise_list[0]](p).matrix()
        kraus = E
        l = len(noise_list)
        for q in n_qubits[::l]:
            for i in range(l):
                noise_op = noise_op_map[noise_list[i]]
                matrices = noise_op(p).matrix()
                circuit += noise_op(p).on(q + i)
                new_kraus = []
                if not (q == 0 and i == 0):
                    for m in kraus:
                        for e in matrices:
                            new_kraus.append(np.kron(m, e))
                    kraus = new_kraus

        # for q in n_qubits[::3]:
            # circuit += BitFlipChannel(p).on(q)
            # new_kraus = []
            # if q != 0:
            #     for m in kraus:
            #         for e in BitFlipChannel(p).matrix():
            #             new_kraus.append(np.kron(m, e))
            #     kraus = new_kraus
            #
            # new_kraus = []
            # if q + 1 < circuit.n_qubits and l > 1:
            #     circuit += DepolarizingChannel(p).on(q + 1)
            #     for m in kraus:
            #         for e in DepolarizingChannel(p).matrix():
            #             new_kraus.append(np.kron(m, e))
            #     kraus = new_kraus
            #
            # new_kraus = []
            # if q + 2 < circuit.n_qubits and l > 2:
            #     circuit += PhaseFlipChannel(p).on(q + 2)
            #     for m in kraus:
            #         for e in PhaseFlipChannel(p).matrix():
            #             new_kraus.append(np.kron(m, e))
            #     kraus = new_kraus
    elif noise == "custom":
        noise_ = noise
        data = load(kraus_file)
        kraus = data['kraus']
        for i in range(kraus.shape[0]):
            if kraus[i].shape[0] != circuit.n_qubits or kraus[i].shape[1] != circuit.n_qubits:
                raise RuntimeError("The dimension of the kraus operator is {}, not consistent with "
                                   "the circuit's ({}, {})! ".format(kraus[i].shape, 2**circuit.n_qubits, 2**circuit.n_qubits))
    else:
        noise_op = noise_op_map[noise]
        noise_ = noise_op.__name__
        noise_ = noise_[0: noise_.index("Channel")]
        E = noise_op(p).matrix()
        # print(E)
        kraus = E
        if noise_ == "Depolarizing":
            for q in n_qubits[::2]:
                circuit += noise_op(p).on(q)
                new_kraus = []
                if q != 0:
                    print(q + 1)
                    print(len(kraus))
                    for i in kraus:
                        for e in E:
                            new_kraus.append(np.kron(i, e))
                    kraus = new_kraus
                new_kraus = []
                for i in kraus:
                    new_kraus.append(np.kron(i, I))
                kraus = new_kraus

        else:
            for q in n_qubits:
                circuit += noise_op(p).on(q)
                # get all kraus operators
                if q != 0:
                    # print(q + 1)
                    # print(len(kraus))
                    new_kraus = []
                    for i in kraus:
                        for e in E:
                            new_kraus.append(np.kron(i, e))
                    kraus = new_kraus

    print(len(kraus))
    # print(kraus[0].shape)
    for i in range(len(kraus)):
        # print(kraus[i].shape)
        # print(kraus[i].shape[0])
        # a = kraus[i]
        # b = U
        # def multi(i):
        #     return np.dot(a[i * 16: (i + 1) * 16, :], b)
        #
        # pool = Pool(4)  # 使用8个进程
        # result = pool.map(multi, range(int(kraus[i].shape[0]/16)))
        # kraus[i] = np.concatenate(result)
        kraus[i] = kraus[i] @ U

    print("add {} with probability {}".format(noise, p))

    for m in all_measures:
        circuit += m

    model_name = "{}_with_{}_{}_model.svg".format(file[file.rfind('/') + 1:-5], p, noise_)
    circuit.svg().to_file("./Figures/" + model_name)  # qasm_file chop '.qasm'
    print(model_name + " saved successfully! ")

    kraus = np.array(kraus)
    print(kraus.shape)
    return kraus


def qasm2mq_with_random_noise(file):
    f = open(file)
    qasm_str = f.read()
    f.close()
    circuit = OpenQASM().from_string(qasm_str)
    if circuit.parameterized:
        val_list = []
        for param in circuit.params_name:
            param = param.replace('pi', str(np.pi)).replace('π', str(np.pi))
            val_list.append(float(param))
        pr = dict(zip(circuit.params_name, val_list))  # 获取线路参数
        circuit = circuit.apply_value(pr)

    all_measures = []
    for gate in circuit:
        if type(gate) == Measure:
            all_measures.append(gate)

    if circuit.has_measure_gate:
        circuit = circuit.remove_measure()
    U = circuit.matrix()

    # add random noise
    noise_op = choice(noise_op_mq)
    p = float(round(uniform(0, 0.2), 5))  # 随机数的精度round(数值，精度)
    noise = noise_op.__name__
    print("add {} with probability {}".format(noise, p))
    for q in range(circuit.n_qubits):
        circuit += noise_op(p).on(q)

    for m in all_measures:
        circuit += m

    noise = noise[0:noise.index("Channel")]
    model_name = "{}_with_{}_{}_model.svg".format(file[file.rfind('/') + 1:-5], p, noise)
    # circuit.svg().to_file("./Figures/" + model_name)  # qasm_file chop '.qasm'
    print(model_name + " saved successfully! ")

    # get all kraus operators
    E = noise_op(p).matrix()
    kraus = E
    for i in range(circuit.n_qubits - 1):
        new_kraus = []
        for m in kraus:
            for e in E:
                new_kraus.append(np.kron(m, e))
        kraus = new_kraus

    for i in range(len(kraus)):
        kraus[i] = kraus[i] @ U

    kraus = np.array(kraus)
    print(kraus.shape)
    return kraus, p, noise


digits = '36'
ADVERSARY_EXAMPLE = False
# noise_type = ''
p = 0

if '.npz' in str(argv[1]):
    # for example:
    # python3 batch_check.py binary_cav.npz 0.001 1 mixed
    data_file = str(argv[1])
    eps = float(argv[2])
    n = int(argv[3])
    state_flag = str(argv[4])

    DATA = load(data_file)
    kraus = DATA['kraus']
    O = DATA['O']
    data = DATA['data']
    label = DATA['label']
    type = 'npz'
    file_name = '{}_{}_{}_{}.csv'.format(
        data_file[data_file.rfind('/') + 1: data_file.rfind('_')], eps, n, state_flag)  # 默认文件名
else:
    # '.qasm' in str(argv[1])
    # for example:
    # python batch_check.py ./model_and_data/mnist56.qasm ./model_and_data/mnist56_data.npz 0.001 1 pure true (argv[6])
    # python batch_check.py ./model_and_data/mnist56.qasm ./model_and_data/mnist56_data.npz 0.001 1 pure true phase_flip 0.001 (argv[8])
    # python batch_check.py ./model_and_data/FashionMNIST.qasm ./model_and_data/FashionMNIST_data.npz 0.001 1 pure (argv[5])
    # python batch_check.py ./model_and_data/FashionMNIST.qasm ./model_and_data/FashionMNIST_data.npz 0.001 1 pure phase_flip 0.001 (argv[7])
    # python batch_check.py ./model_and_data/iris.qasm ./model_and_data/iris_data.npz 0.001 1 mixed (argv[5])
    # python batch_check.py ./model_and_data/iris.qasm ./model_and_data/iris_data.npz 0.001 1 mixed phase_flip 0.001 (argv[7])
    qasm_file = str(argv[1])
    data_file = str(argv[2])
    eps = float(argv[3])
    n = int(argv[4])
    state_flag = str(argv[5])

    if 'mnist' in data_file:  # digits != '36'
        ADVERSARY_EXAMPLE = (str(argv[6]) == 'true')
        if '_data' in data_file:  # digits != '36'
            digits = data_file[data_file.rfind('_data') - 2: data_file.rfind('_data')]

    if len(argv) > 7:
        # noise_type = argv[len(argv) - 2]
        noise_type = argv[7]
        p = float(argv[len(argv) - 1])
        noise_list = []
        kraus_file = None
        if noise_type == 'mixed':
            noise_list = [i for i in argv[8: len(argv) - 1]]
            print("noise_list: ", noise_list)
        elif noise_type == 'custom':
            kraus_file = argv[8]
        kraus = qasm2mq_with_specified_noise(qasm_file, noise_type, noise_list, kraus_file, p)
    else:
        kraus, p, noise_type = qasm2mq_with_random_noise(qasm_file)
    # kraus = qasm2mq(qasm_file)
    DATA = load(data_file)
    O = DATA['O']
    data = DATA['data']
    label = DATA['label']
    type = 'qasm'
    file_name = '{}_{}_{}_{}_{}_{}.csv'.format(
        qasm_file[qasm_file.rfind('/') + 1:-5], eps, n, state_flag, p, noise_type)  # 默认文件名
    # file_name = '{}_{}_{}_{}.csv'.format(
    #     qasm_file[qasm_file.rfind('/') + 1:-5], eps, n, state_flag)  # 默认文件名

if state_flag == 'mixed':
    verifier = RobustnessVerifier
else:
    verifier = PureRobustnessVerifier

ac = PrettyTable()
time = PrettyTable()
ac.add_column('epsilon', ['Robust Bound', 'Robustness Algorithm'])
time.add_column('epsilon', ['Robust Bound', 'Robustness Algorithm'])
for j in range(n):
    c_eps = eps * (j + 1)
    if 'mnist' in data_file:
        ac_temp, time_temp = verifier(kraus, O, data, label, c_eps, type, ADVERSARY_EXAMPLE, digits, 'mnist')
    else:
        ac_temp, time_temp = verifier(kraus, O, data, label, c_eps, type)

    ac.add_column('{:e}'.format(c_eps), [
        '{:.2f}'.format(ac_temp[0] * 100),
        '{:.2f}'.format(ac_temp[1] * 100)])
    time.add_column('{:e}'.format(c_eps), [
        '{:.4f}'.format(time_temp[0]),
        '{:.4f}'.format(time_temp[1])])

file_path = './results/result_tables/' + file_name
# print(file_path)

with open(file_path, 'w', newline='') as f_output:
    f_output.write(ac.get_csv_string())
    f_output.write('\n')
    f_output.write(time.get_csv_string())
    f_output.close()
    print(file_name + " saved successfully! ")


# ac_1 = []
# ac_2 = []
# time_1 = []
# time_2 = []
# model_name = data_file[data_file.rfind('/') + 1: data_file.rfind('_')]
# noise_name = noise_type.replace('_', ' ')
# with open("./results/local_results.csv", "a+") as csvfile_ac:
#     # with open("./results/{}_results.csv".format(model_name), "a+") as csvfile_ac:
#     # with open("./results/time_{}.csv".format(model_name), "a+") as csvfile_time:
#     w_ac = csv.writer(csvfile_ac)
#     # w_time = csv.writer(csvfile_time)
#     for j in range(n):
#         c_eps = eps * (j + 1)
#         if 'mnist' in data_file:
#             ac_temp, time_temp = verifier(kraus, O, data, label, c_eps, type, ADVERSARY_EXAMPLE, digits, 'mnist')
#         else:
#             ac_temp, time_temp = verifier(kraus, O, data, label, c_eps, type)
#
#         # ac_1.append(np.round(ac_temp[0] * 100, 2))
#         # ac_2.append(np.round(ac_temp[1] * 100, 2))
#         # time_1.append(np.round(time_temp[0], 4))
#         # time_2.append(np.round(time_temp[1], 4))
#         ac_1 = np.round(ac_temp[0] * 100, 2)
#         ac_2 = np.round(ac_temp[1] * 100, 2)
#         time_1 = np.round(time_temp[0], 4)
#         time_2 = np.round(time_temp[1], 4)
#         w_ac.writerow([model_name, noise_name, p, c_eps, ac_1, time_1, ac_2, time_2])
#         # 逐行写入数据 (写入多行用writerows)
#         # w_time.writerow([model_name, noise_name, p] + time_1 + time_2)


def file_test():
    model_name = qasm_file[qasm_file.rfind('/') + 1:-5]
    print(model_name)
    probs = [0.01, 0.001]
    with open("./results/ac_{}.csv".format(model_name), "a+") as csvfile_ac:
        with open("./results/time_{}.csv".format(model_name), "a+") as csvfile_time:
            w_ac = csv.writer(csvfile_ac)
            w_time = csv.writer(csvfile_time)
            w_ac.writerow(['Noise Type', 'Probability',
                           'ε=0.001', 'ε=0.002', 'ε=0.003', 'ε=0.004 (Robust Bound)',
                           'ε=0.001', 'ε=0.002', 'ε=0.003', 'ε=0.004 (Robustness Algorithm)'])
            w_time.writerow(['Noise Type', 'Probability',
                             'ε=0.001', 'ε=0.002', 'ε=0.003', 'ε=0.004 (Robust Bound)',
                             'ε=0.001', 'ε=0.002', 'ε=0.003', 'ε=0.004 (Robustness Algorithm)'])
            for noise_type in noise_op_map:
                noise_name = noise_type.replace('_', ' ')
                for p in probs:
                    k = qasm2mq_with_specified_noise(qasm_file, noise_type, p)
                    ac_1 = []
                    ac_2 = []
                    time_1 = []
                    time_2 = []
                    for j in range(n):
                        c_eps = eps * (j + 1)
                        if 'mnist' in data_file:
                            ac_temp, time_temp = verifier(k, O, data, label, c_eps, type, ADVERSARY_EXAMPLE, digits,
                                                          'mnist')
                        else:
                            ac_temp, time_temp = verifier(k, O, data, label, c_eps, type)

                        ac_1.append(np.round(ac_temp[0] * 100, 2))
                        ac_2.append(np.round(ac_temp[1] * 100, 2))
                        time_1.append(np.round(time_temp[0], 4))
                        time_2.append(np.round(time_temp[1], 4))

                    # 逐行写入数据 (写入多行用writerows)
                    w_ac.writerow(
                        [noise_name, p, ac_1[0], ac_1[1], ac_1[2], ac_1[3], ac_2[0], ac_2[1], ac_2[2], ac_2[3]])
                    w_time.writerow(
                        [noise_name, p, time_1[0], time_1[1], time_1[2], time_1[3], time_2[0], time_2[1], time_2[2],
                         time_2[3]])
                    # gc.collect()

# file_test()
