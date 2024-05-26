import numpy as np
from numpy import load
import random

from mindquantum.io import OpenQASM
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import BitFlipChannel, DepolarizingChannel, PhaseFlipChannel, Measure, I
from mindquantum.core.gates.basic import BasicGate

noise_ops = ["phase_flip", "depolarizing", "bit_flip"]

noise_op_map = {
    "bit_flip": BitFlipChannel,
    "depolarizing": DepolarizingChannel,
    "phase_flip": PhaseFlipChannel,
}

noise_name_map = {
    'bit_flip': 'BitFlip',
    'depolarizing': 'Depolarizing',
    'phase_flip': 'PhaseFlip',
    'mixed': 'mixed_BitFlip_Depolarizing_PhaseFlip'
}

noise_name_map_reverse = {
    'BitFlip': 'bit_flip',
    'Depolarizing': 'depolarizing',
    'PhaseFlip': 'phase_flip',
    # 'mixed_BitFlip_Depolarizing_PhaseFlip': 'mixed'
}

I_matrix = I.matrix()


def qasm2mq(qasm_file, to_save_figure=False, filepath=''):
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

    if to_save_figure:
        if filepath != '':
            circuit.svg().to_file(filepath)  # qasm_file chop '.qasm'
            print(filepath + "was saved successfully! ")
        else:
            model_name = "{}_origin.svg".format(qasm_file[qasm_file.rfind('/') + 1:-5])
            circuit.svg().to_file("./figures/" + model_name)  # qasm_file chop '.qasm'
            print(model_name + " was saved successfully! ")

    # if circuit.has_measure_gate:
    #     circuit = circuit.remove_measure()
    #
    # U = circuit.matrix()
    # kraus = np.array([U])
    return circuit


def qasm2mq_with_kraus(qasm_file, to_save_figure=False, filepath=''):
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

    if to_save_figure:
        if filepath != '':
            circuit.svg().to_file(filepath)  # qasm_file chop '.qasm'
            print(filepath + " was saved successfully! ")
        else:
            model_name = "{}_origin.svg".format(qasm_file[qasm_file.rfind('/') + 1:-5])
            circuit.svg().to_file("./figures/" + model_name)  # qasm_file chop '.qasm'
            print(model_name + " was saved successfully! ")

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
                            # print('len(selected_qubits) =', len(selected_qubits))
                            # print('kraus_.shape =', np.array(kraus_).shape)
        final_circuit.append(random_circit)
        print('random_kraus.shape =', np.array(kraus_).shape)
    return Circuit(final_circuit[0]), np.array(kraus_)


def generating_circuit_with_random_noise(circ, model_name_, to_save_figure=False, filepath=''):
    # generate random noise
    # noise_num = random.randint(1, len(circ))
    noise_num = circ.n_qubits
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

    print('add {} noise: {}'.format(noise_num, ops))

    # insert random noise
    circ, kraus_ = random_insert_ops(circ, ops)

    for m in all_measures:
        circ += m

    if to_save_figure:
        if filepath != '':
            circ.svg().to_file(filepath)  # qasm_file chop '.qasm'
            print(filepath + "was saved successfully! ")
        else:
            file_name_ = '{}_random.svg'.format(model_name_)
            circ.svg().to_file("./figures/" + file_name_)  # qasm_file chop '.qasm'
            print(file_name_ + " was saved successfully! ")

    return circ, kraus_


def generating_circuit_with_specified_noise(circ, origin_kraus_, noise, noise_list_, kraus_file_, noise_p_: float,
                                            model_name_, to_save_figure=False, filepath=''):
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
        dim = 2 ** circ.n_qubits
        for i in range(kraus_.shape[0]):
            if kraus_[i].shape[0] != dim or kraus_[i].shape[1] != dim:
                raise RuntimeError("The dimension of the kraus operator is {}, not consistent with "
                                   "the circuit's ({}, {})! ".format(kraus_[i].shape, dim, dim))
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

    if to_save_figure:
        if filepath != '':
            circ.svg().to_file(filepath)  # qasm_file chop '.qasm'
            print(filepath + "was saved successfully! ")
        else:
            file_name_ = '{}_{}_{}.svg'.format(model_name_, noise_name_, noise_p_)
            circ.svg().to_file("./figures/" + file_name_)  # qasm_file chop '.qasm'
            print(file_name_ + " was saved successfully! ")

    new_kraus_ = np.array(new_kraus_)
    print('final_kraus.shape = {}\n'.format(new_kraus_.shape))
    return new_kraus_, noise_name_
