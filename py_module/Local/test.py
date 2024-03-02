from VeriQ import RobustnessVerifier, PureRobustnessVerifier
from sys import argv
import numpy as np
from numpy import load

from mindquantum.io import OpenQASM
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import BitFlipChannel, DepolarizingChannel, PhaseFlipChannel, Measure
from random import choice, uniform
import mindquantum
import csv

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

    model_name = "{}_model.svg".format(qasm_file[qasm_file.rfind('/') + 1:-5])
    circuit.svg().to_file("./figures/" + model_name)  # qasm_file chop '.qasm'
    print(model_name + " saved successfully! ")

    if circuit.has_measure_gate:
        circuit = circuit.remove_measure()

    U = circuit.matrix()
    kraus = np.array([U])
    return kraus


I = mindquantum.core.gates.I.matrix()


def qasm2mq_with_specified_noise(file, noise_type, p: float):
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
        if type(gate) == Measure:
            all_measures.append(gate)

    if circuit.has_measure_gate:
        circuit = circuit.remove_measure()
    U = circuit.matrix()

    # add random noise
    num = circuit.n_qubits
    n_qubits = range(num)
    if noise_type == "mixed":
        noise = noise_type
        # get all kraus operators
        E = BitFlipChannel(p).matrix()
        kraus = E
        for q in n_qubits[::3]:
            circuit += BitFlipChannel(p).on(q)
            new_kraus = []
            if q != 0:
                for m in kraus:
                    for e in BitFlipChannel(p).matrix():
                        new_kraus.append(np.kron(m, e))
                kraus = new_kraus

            new_kraus = []
            if q + 1 < circuit.n_qubits:
                circuit += DepolarizingChannel(p).on(q + 1)
                for m in kraus:
                    for e in DepolarizingChannel(p).matrix():
                        new_kraus.append(np.kron(m, e))
                kraus = new_kraus

            new_kraus = []
            if q + 2 < circuit.n_qubits:
                circuit += PhaseFlipChannel(p).on(q + 2)
                for m in kraus:
                    for e in PhaseFlipChannel(p).matrix():
                        new_kraus.append(np.kron(m, e))
                kraus = new_kraus
    else:
        noise_op = noise_op_map[noise_type]
        noise = noise_op.__name__
        noise = noise[0: noise.index("Channel")]
        E = noise_op(p).matrix()
        # print(E)
        kraus = E
        if noise == "Depolarizing":
            for q in n_qubits[::2]:
                circuit += noise_op(p).on(q)
                new_kraus = []
                if q != 0:
                    # print(q + 1)
                    # print(len(kraus))
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
    for i in range(len(kraus)):
        kraus[i] = kraus[i] @ U

    print("add {} with probability {}".format(noise_type, p))

    for m in all_measures:
        circuit += m

    # model_name = "{}_with_{}_{}_model.svg".format(file[file.rfind('/') + 1:-5], p, noise)
    # circuit.svg().to_file("./figures/" + model_name)  # qasm_file chop '.qasm'
    # print(model_name + " saved successfully! ")

    kraus = np.array(kraus)
    print(kraus.shape)
    return kraus


n = 1
state_flag = 'pure'
ADVERSARY_EXAMPLE = True

if state_flag == 'mixed':
    verifier = RobustnessVerifier
else:
    verifier = PureRobustnessVerifier

# for d0 in range(1, 10):
#     for d1 in range(d0+1, 10):
#         digits = str(d0) + str(d1)
        # if digits == '49':
        #     continue
digits = '09'
data_file = './model_and_data/mnist{}_data.npz'.format(digits)
# data_file = './model_and_data/TFIchain8_data.npz'
model_name = data_file[data_file.rfind('/') + 1: data_file.rfind('_')]
DATA = load(data_file)
O = DATA['O']
data = DATA['data']
label = DATA['label']

noise_type = choice(["bit_flip", "depolarizing", "phase_flip", "mixed"])
p = choice([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075])
eps = choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01])
# eps = choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075])

# noise_type = "mixed"
# p = 0.05
# eps = 0.01

print('noise_type =', noise_type)
print('p =', p)
print('eps =', eps)

qasm_file = './model_and_data/' + model_name + '.qasm'
kraus = qasm2mq_with_specified_noise(qasm_file, noise_type, p)
type = 'qasm'

noise_name = noise_type.replace('_', ' ')
with open("./results/local_results.csv", "a+") as csvfile:
    w = csv.writer(csvfile)
    for j in range(n):
        c_eps = eps * (j + 1)
        if 'mnist' in model_name:
            ac_temp, time_temp = verifier(kraus, O, data, label, c_eps, type, ADVERSARY_EXAMPLE, digits, 'mnist')
        else:
            ac_temp, time_temp = verifier(kraus, O, data, label, c_eps, type)

        ac_1 = np.round(ac_temp[0] * 100, 2)
        ac_2 = np.round(ac_temp[1] * 100, 2)
        time_1 = np.round(time_temp[0], 4)
        time_2 = np.round(time_temp[1], 4)
        w.writerow([model_name, noise_name, p, c_eps, ac_1, time_1, ac_2, time_2])
