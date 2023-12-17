from VeriQ import RobustnessVerifier, PureRobustnessVerifier
from prettytable import PrettyTable
from sys import argv
import numpy as np
from numpy import load

from mindquantum.io import OpenQASM
from mindquantum.core.gates import BitFlipChannel, DepolarizingChannel, PhaseFlipChannel
from random import choice, uniform


noise_op = [
    PhaseFlipChannel,
    DepolarizingChannel,
    BitFlipChannel,
    DepolarizingChannel
]


def qasm2mq_with_random_noise(qasm_file):
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

    kraus, p, noise_name = qasm2mq_with_random_noise(qasm_file)
    DATA = load(data_file)
    O = DATA['O']
    data = DATA['data']
    label = DATA['label']
    type = 'qasm'
    file_name = '{}_{}_{}_{}_{}_{}.csv'.format(
        qasm_file[qasm_file.rfind('/')+1:-5], eps, n, state_flag, p, noise_name)  # 默认文件名

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
        flag = str(argv[5])
        ADVERSARY_EXAMPLE = True if flag == 'true' else False
        ac_temp, time_temp = verifier(kraus, O, data, label, c_eps, type, ADVERSARY_EXAMPLE)
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
