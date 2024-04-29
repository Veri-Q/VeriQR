from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import BitFlipChannel, DepolarizingChannel, PhaseFlipChannel, Measure
from mindquantum.io import OpenQASM
import numpy as np
import random
from cirq import circuits
import csv


def qasm2mq(qasm_file):
    f = open(qasm_file)
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
    return circuit


def random_insert_ops(circuit, nums_and_ops, with_ctrl=True, after_measure=False, shots=1):
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

    circuit = circuit.remove_barrier()  # 去掉栅栏门
    available_indexs = []
    if after_measure:
        available_indexs = range(len(circuit))
    else:
        for i, gate in enumerate(circuit):
            if not isinstance(gate, Measure):
                available_indexs.append(i)

    final_circuit = []
    for _ in range(shots):
        nums, ops = [], []
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
            if not isinstance(gate, Measure) or after_measure:
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
        final_circuit.append(circ)
    return Circuit(final_circuit[0])


# noise_ops = ["phase_flip", "depolarizing", "bit_flip"]
# noise_op_map = {
#     "bit_flip": BitFlipChannel,
#     "depolarizing": DepolarizingChannel,
#     "phase_flip": PhaseFlipChannel,
#     # "mixed": DepolarizingChannel
# }
#
# circ = qasm2mq('./model_and_data/iris.qasm')
# circ.svg().to_file('./iris_origin.svg')
#
# # random noise
# noise_num = random.randint(1, len(circ))
# print('add {} noise'.format(noise_num))
# ops = []
# left_noise_num = noise_num
# while left_noise_num > 0:
#     noise = noise_op_map[random.choice(noise_ops)]
#     print(noise.__name__)
#     p = float(round(random.random(), 5))  # 随机数的精度round(数值，精度)
#     noise_op = noise(p)
#     temp_noise_num = random.randint(0, left_noise_num)
#     # print('temp_noise_num =', temp_noise_num)
#     if temp_noise_num != 0:
#         ops.append([temp_noise_num, noise_op])
#         left_noise_num -= temp_noise_num
#         # print('[{}, {}]'.format(temp_noise_num, noise_op))
#         # print('left_noise_num =', left_noise_num)
#
# circ = random_insert_ops(circ, ops)
# circ.svg().to_file('./iris_noisy.svg')


# DATA = np.load('./model_and_data/fashion8_data.npz')
# O = DATA['O']
# data = DATA['data']
# label = DATA['label']
# print(data.shape)
# print(label.shape)


with open("./results/local_results_v2.csv") as f:
    with open("./results/local_results.csv", 'a+') as csvfile:
        w = csv.writer(csvfile)
        for row in csv.reader(f, skipinitialspace=True):
            if row == [] or row[0] == 'Model' or len(row) < 16:
                continue
            model_name = row[0]
            noise_type = row[1].replace(' ', '-')
            noise_p = row[2]
            c_eps = row[3]
            w.writerows([
                [model_name, '{}_{}_{}'.format(noise_type, noise_p, c_eps), 'c_0',
                 '%.2f' % float(row[4]), '%.4f' % float(row[5]), '%.2f' % float(row[6]), '%.4f' % float(row[7])],
                [model_name, '{}_{}_{}'.format(noise_type, noise_p, c_eps), 'c_1',
                 '%.2f' % float(row[8]), '%.4f' % float(row[9]), '%.2f' % float(row[10]), '%.4f' % float(row[11])],
                [model_name, '{}_{}_{}'.format(noise_type, noise_p, c_eps), 'c_2',
                 '%.2f' % float(row[12]), '%.4f' % float(row[13]), '%.2f' % float(row[14]), '%.4f' % float(row[15])]
            ])
