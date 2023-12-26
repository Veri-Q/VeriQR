from VeriQ import RobustnessVerifier, PureRobustnessVerifier
from prettytable import PrettyTable
from sys import argv
import numpy as np
from numpy import load

from mindquantum.io import OpenQASM
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import BitFlipChannel, DepolarizingChannel, PhaseFlipChannel, Measure
from random import choice, uniform


noise_op = [
    PhaseFlipChannel,
    DepolarizingChannel,
    BitFlipChannel,
    DepolarizingChannel
]


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

def qasm2mq_with_random_noise(qasm_file):
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

    model_name = "{}_model.svg".format(qasm_file[qasm_file.rfind('/') + 1:-5])
    circuit.svg().to_file("./Figures/" + model_name)  # qasm_file chop '.qasm'
    print(model_name + " saved successfully! ")

    U = circuit.matrix()
    kraus = np.array([U])
    return kraus



def qasm2mq_with_random_noise(qasm_file):
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

    circ_ = circuit
    if circ_.has_measure_gate:
        circ_ = circ_.remove_measure()
    U = circ_.matrix()

    # add random noise
    noise_op_mq = choice(noise_op)
    p = float(round(uniform(0, 0.2), 5))  # 随机数的精度round(数值，精度)
    noise_name = noise_op_mq.__name__
    print("add {} with probability {}".format(noise_name, p))
    for q in range(circuit.n_qubits):
        circuit += noise_op_mq(p).on(q)

    noise_name = noise_name[0:noise_name.index("Channel")]
    model_name = "{}_with_{}_{}_model.svg".format(qasm_file[qasm_file.rfind('/')+1:-5], p, noise_name)
    circuit.svg().to_file("./Figures/" + model_name)  # qasm_file chop '.qasm'
    print(model_name + " saved successfully! ")

    # get all kraus operators
    E = noise_op_mq(p).matrix()
    kraus = E
    for i in range(circuit.n_qubits-1):
        new_kraus = []
        for m in kraus:
            for e in E:
                new_kraus.append(np.kron(m, e))
        kraus = new_kraus

    for i in range(len(kraus)):
        kraus[i] = kraus[i] @ U

    kraus = np.array(kraus)
    # print(kraus.shape)
    return kraus, p, noise_name


def qasm2mq(qasm_file):
    f = open(qasm_file)
    qasm = f.read()
    f.close()
    circuit = OpenQASM().from_string(qasm)
    # print(circuit)
    # print(circuit.parameterized)
    # print(circuit.params_name)
    if circuit.parameterized:
        val_list = []
        for param in circuit.params_name:
            param = param.replace('pi', str(np.pi)).replace('π', str(np.pi))
            val_list.append(float(param))
        pr = dict(zip(circuit.params_name, val_list))  # 获取线路参数
        circuit = circuit.apply_value(pr)

    circuit.svg().to_file(
        "./Figures/" + qasm_file[qasm_file.rfind('/') + 1:-5] + "_model.svg")  # qasm_file chop '.qasm'

    if circuit.has_measure_gate:
        circuit = circuit.remove_measure()

    mat = circuit.matrix()
    kraus = np.array([mat])
    return kraus


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
    # python3 batch_check.py iris.qasm iris_data.npz 0.001 1 mixed
    qasm_file = str(argv[1])
    data_file = str(argv[2])
    eps = float(argv[3])
    n = int(argv[4])
    state_flag = str(argv[5])

    # kraus, p, noise_name = qasm2mq_with_random_noise(qasm_file)
    kraus = qasm2mq(qasm_file)
    DATA = load(data_file)
    O = DATA['O']
    data = DATA['data']
    label = DATA['label']
    type = 'qasm'
    # file_name = '{}_{}_{}_{}_{}_{}.csv'.format(
    #     qasm_file[qasm_file.rfind('/')+1:-5], eps, n, state_flag, p, noise_name)  # 默认文件名
    file_name = '{}_{}_{}_{}.csv'.format(
        qasm_file[qasm_file.rfind('/') + 1:-5], eps, n, state_flag)  # 默认文件名

if state_flag == 'mixed':
    verifier = RobustnessVerifier
else:
    verifier = PureRobustnessVerifier

digits = '36'
ADVERSARY_EXAMPLE = False
if 'mnist' in data_file:
    if '_data' in data_file:
        flag = str(argv[6])
        digits = data_file[data_file.rfind('_data')-2: data_file.rfind('_data')]
    else:
        flag = str(argv[5])
        # digits = '36'
    ADVERSARY_EXAMPLE = (flag=='true')

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

# print('Robust Accuracy (in Percent)')
# print(ac)
# print('Verification Times (in Seconds)')
# print(time)


file_path = './results/result_tables/' + file_name
# print(file_name)
# print(file_path)

with open(file_path, 'w', newline='') as f_output:
    f_output.write(ac.get_csv_string())
    f_output.write('\n')
    f_output.write(time.get_csv_string())
    f_output.close()
    print(file_name + " saved successfully! ")
