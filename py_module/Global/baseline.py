import os
import csv
from common_interface import *


jax.config.update('jax_platform_name', 'cpu')
tn.set_default_backend("jax")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


# def random_insert_ops(mq_circuit: mindquantum.Circuit, cirq_circuit: cirq.Circuit,
#                       mq_nums_and_ops, cirq_nums_and_ops, with_ctrl=True, after_measure=False):
#     """Insert single-qubit operators at random points in the circuit.
#
#     args:
#         circuit: The quantum circuit in which the operators are to be inserted.
#         nums_and_ops: [num_0, op_0], [num_1, op_1], ...
#             Where num_i is the number of insertion operators op_i. num_i: int, op_i: gate.
#         with_ctrl: Whether to allow operator insertion on control qubits.
#         after_measure: Whether to allow insertion of an operator after a measurement gate.
#         shots: The number of new circuits generated.
#
#     returns:
#         An iterator of the new circuit generated after inserting the operators.
#     """
#     print(mq_nums_and_ops)
#     print(cirq_nums_and_ops)
#
#     mq_circuit = mq_circuit.remove_barrier()
#     available_indexes = []
#     if after_measure:
#         available_indexes = range(len(mq_circuit))
#     else:
#         for i, mq_gate in enumerate(mq_circuit):
#             if not isinstance(mq_gate, Measure):
#                 available_indexes.append(i)
#
#     mq_nums, mq_ops = [], []
#     cirq_nums, cirq_ops = [], []
#     for i in range(len(mq_nums_and_ops)):
#         if len(mq_nums_and_ops[i]) != 2:
#             raise ValueError(
#                 f'The format of the argment "nums_and_ops" should be "[num_0, op_0], [num_1, op_1], ....".')
#         if mq_nums_and_ops[i][0] > len(available_indexes):
#             raise ValueError(
#                 f'The number of positions allowed to insert channel should be less than {len(available_indexes)}, but get {mq_nums_and_ops[i][0]}.')
#         mq_nums.append(mq_nums_and_ops[i][0])
#         mq_ops.append(mq_nums_and_ops[i][1])
#         cirq_nums.append(cirq_nums_and_ops[i][0])
#         cirq_ops.append(cirq_nums_and_ops[i][1])
#     indexes = []
#     for num in mq_nums:
#         tem = sorted(np.random.choice(available_indexes, size=num, replace=False))
#         indexes.append(tem)
#
#     mq_random_circuit = Circuit()
#     cirq_random_circuit = cirq.Circuit()
#     qubits_num = mq_circuit.n_qubits
#     # cirq_qubits = cirq.GridQubit.rect(1, qubits_num)
#     # cirq_qubits = cirq.num_qubits(qubits_num)
#     cirq_qubits = sorted(cirq_circuit.all_qubits())
#     selected_qubits = []
#     # U1 = cirq.unitary(cirq_circuit)
#     # M = U1.conj().T @ np.kron(np.eye(2 ** (qubits_num - 1)), np.array([[1., 0.], [0., 0.]])) @ U1
#     for (i, mq_gate), cirq_gate in zip(enumerate(mq_circuit), cirq_circuit.all_operations()):
#         if ((isinstance(mq_gate, Measure) or isinstance(cirq_gate, Measure))
#                 and not after_measure):
#             continue
#
#         mq_random_circuit += mq_gate
#         cirq_random_circuit += cirq_gate
#         for j, tem_indexs in enumerate(indexes):
#             for k in tem_indexs:
#                 if k == i:
#                     qubits = (mq_gate.ctrl_qubits + mq_gate.obj_qubits) if with_ctrl else mq_gate.obj_qubits
#                     qubit = int(np.random.choice(qubits))
#                     if qubit in selected_qubits:  # the `qubit` has been selected.
#                         continue
#                     # mq_noise_op = mq_ops[j]
#                     mq_random_circuit += mq_ops[j].on(qubit)
#                     cirq_random_circuit += cirq_ops[j].on(cirq_qubits[qubit])
#                     # kraus = cirq_ops[j](cirq_qubits[qubit])._mixture_()
#                     # N = 0
#                     # for E in kraus:
#                     #     F = np.kron(np.eye(2 ** qubit), np.kron(E[1], np.eye(2 ** (qubits_num - qubit - 1))))
#                     #     N += E[0] * F.conj().T @ M @ F
#                     # M = N
#                     selected_qubits.append(qubit)
#
#     return mq_random_circuit, cirq_random_circuit


# def generating_circuit_with_random_noise(mq_circ: mindquantum.Circuit, cirq_circ: cirq.Circuit, model_name_):
#     # generate random noise
#     # noise_num = random.randint(1, len(circ))
#     noise_num = mq_circ.n_qubits
#     print('add {} noise'.format(noise_num))
#     mq_ops, cirq_ops = [], []
#     left_noise_num = noise_num
#     while left_noise_num > 0:
#         noise_p_ = float(round(random.uniform(0, 0.2), 5))
#         noise_type_ = random.choice(noise_ops)
#         mq_noise_op = noise_op_mq[noise_type_](noise_p_)
#         cirq_noise_op = noise_op_cirq[noise_type_](noise_p_)
#         temp_noise_num = random.randint(0, left_noise_num)
#         if temp_noise_num != 0:
#             mq_ops.append([temp_noise_num, mq_noise_op])
#             cirq_ops.append([temp_noise_num, cirq_noise_op])
#             left_noise_num -= temp_noise_num
#             # print('[{}, {}]'.format(temp_noise_num, noise_op))
#             # print('left_noise_num =', left_noise_num)
#
#     # remove measures
#     all_measures = []
#     for gate in mq_circ:
#         if isinstance(gate, Measure):
#             all_measures.append(gate)
#     mq_circ = mq_circ.remove_measure()
#
#     # insert random noise
#     mq_circ, cirq_circ = random_insert_ops(mq_circ, cirq_circ, mq_ops, cirq_ops)
#
#     # add measures
#     for m in all_measures:
#         mq_circ += m
#
#     # file_name_ = '{}_random.svg'.format(model_name_)
#     # mq_circ.svg().to_file("./figures/{}/{}".format(model_name_, file_name_))  # qasm_file chop '.qasm'
#     # print(file_name_ + " saved successfully! ")
#     return mq_circ, cirq_circ


# def generating_circuit_with_random_noise(mq_circ: mindquantum.Circuit, cirq_circ: cirq.Circuit, model_name_):
#     # generate random noise
#     # noise_num = random.randint(1, len(circ))
#     noise_num = mq_circ.n_qubits
#     print('add {} noise'.format(noise_num))
#     mq_ops, cirq_ops = [], []
#     left_noise_num = noise_num
#     while left_noise_num > 0:
#         noise_p_ = float(round(random.uniform(0, 0.2), 5))
#         noise_type_ = random.choice(noise_ops)
#         mq_noise_op = noise_op_mq[noise_type_](noise_p_)
#         cirq_noise_op = noise_op_cirq[noise_type_](noise_p_)
#         temp_noise_num = random.randint(0, left_noise_num)
#         if temp_noise_num != 0:
#             mq_ops.append([temp_noise_num, mq_noise_op])
#             cirq_ops.append([temp_noise_num, cirq_noise_op])
#             left_noise_num -= temp_noise_num
#             # print('[{}, {}]'.format(temp_noise_num, noise_op))
#             # print('left_noise_num =', left_noise_num)
#
#     # remove measures
#     all_measures = []
#     for gate in mq_circ:
#         if isinstance(gate, Measure):
#             all_measures.append(gate)
#     mq_circ = mq_circ.remove_measure()
#
#     # insert random noise
#     mq_circ, cirq_circ, random_kraus = random_insert_ops(mq_circ, cirq_circ, mq_ops, cirq_ops)
#
#     # add measures
#     for m in all_measures:
#         mq_circ += m
#
#     # file_name_ = '{}_random.svg'.format(model_name_)
#     # mq_circ.svg().to_file("./figures/{}/{}".format(model_name_, file_name_))  # qasm_file chop '.qasm'
#     # print(file_name_ + " saved successfully! ")
#     return mq_circ, cirq_circ, random_kraus


def generating_circuit_with_specified_noise_(mq_circuit: mindquantum.Circuit, cirq_circuit: cirq.Circuit,
                                             noise_type_, noise_list_, kraus_file_, noise_p_,
                                             model_name_):
    all_measures = []
    for gate in mq_circuit:
        # print(type(gate))
        if type(gate) is Measure:
            all_measures.append(gate)
    if mq_circuit.has_measure_gate:
        mq_circuit = mq_circuit.remove_measure()

    # noise_op_cirq_ = noise_op_cirq[noise_type]
    # noise_op_mq_ = noise_op_mq[noise_type]
    qubits_num = mq_circuit.n_qubits
    # qubits = cirq.GridQubit.rect(1, qubits_num)
    qubits = sorted(cirq_circuit.all_qubits())
    # M = random_kraus
    if noise_p_ > 1e-7:
        if noise_type_ == "mixed":
            l = len(noise_list_)
            for q in range(qubits_num)[::l]:
                for i in range(l):
                    if q + i < qubits_num:
                        cirq_circuit += noise_op_cirq[noise_list_[i]](noise_p_).on(qubits[q + i])
                        mq_circuit += noise_op_mq[noise_list_[i]](noise_p_).on(q + i)
                        # kraus = noise_op_cirq[noise_list_[i]](noise_p_)(qubits[q + i])._mixture_()
                        # N = 0
                        # for E in kraus:
                        #     F = np.kron(np.eye(2 ** (q + i)), np.kron(E[1], np.eye(2 ** (qubits_num - (q + i) - 1))))
                        #     N += E[0] * F.conj().T @ M @ F
                        # M = N
            noise_list_ = [noise_op_mq[i].__name__ for i in noise_list_]
            noise_list_ = [i[0: i.index("Channel")] for i in noise_list_]
            noise_name_ = "mixed_{}".format('_'.join(noise_list_))
        elif noise_type_ == "custom":
            # TODO
            data = load(kraus_file_)
            noisy_kraus = data['kraus']
            noise_name_ = "custom_{}".format(kraus_file_[kraus_file_.rfind('/') + 1:-4])
        else:
            # noise = noise_op_cirq[noise_type]
            cirq_circuit += noise_op_cirq[noise_type_](noise_p_).on_each(*qubits)
            for q in range(mq_circuit.n_qubits):
                mq_circuit += noise_op_mq[noise_type_](noise_p_).on(q)
                # kraus = noise_op_cirq[noise_type_](noise_p_)(qubits[q])._mixture_()
                # N = 0
                # for E in kraus:
                #     F = np.kron(np.eye(2 ** q), np.kron(E[1], np.eye(2 ** (qubits_num - q - 1))))
                #     N += E[0] * F.conj().T @ M @ F
                # M = N
            noise_ = noise_op_mq[noise_type_].__name__
            noise_ = noise_[0: noise_.index("Channel")]
            noise_name_ = noise_

    # print("add {} with probability {}".format(noise_type_, noise_p_))
    for m in all_measures:
        mq_circuit += m

    # file_name_ = '{}_{}_{}.svg'.format(model_name_, noise_name_, noise_p_)
    # mq_circuit.svg().to_file("./figures/{}/{}".format(model_name_, file_name_))  # qasm_file chop '.qasm'
    # print(file_name_ + " saved successfully! ")

    print('Circuit: %s' % model_name_)
    print('Noise configuration: {}, {}'.format(noise_name_, noise_p_))
    return mq_circuit, cirq_circuit


# def circuit2M(circuit: cirq.Circuit):
#     qubits_num = len(circuit.all_qubits())
#     U1 = np.kron(np.eye(2 ** (qubits_num - 1)), np.array([[1., 0.], [1., 0.]]))
#     M = U1.conj().T @ np.kron(np.eye(2 ** (qubits_num - 1)), np.array([[1., 0.], [0., 0.]])) @ U1
#     for moment in circuit.moments:
#         try:
#             # unitary
#             U = cirq.unitary(moment)
#             M = U.conj().T @ M @ U
#         except:
#             # print(moment)
#             for op in moment.operations:
#                 try:
#                     # unitary
#                     # op.gate._has_unitary_()
#                     U = cirq.unitary(op.gate)
#                     M = U.conj().T @ M @ U
#                 except:
#                     # noise
#                     kraus = op._mixture_()
#                     q = op.qubits  # TODO
#                     print(op)
#                     print(q)
#                     N = 0
#                     for E in kraus:
#                         F = np.kron(np.eye(2 ** q), np.kron(E[1], np.eye(2 ** (qubits_num - q - 1))))
#                         N += E[0] * F.conj().T @ M @ F
#                     M = N
#     return M

def circuit2matrix(circuit, noise_type, noise_list, kraus_file, p):
    qubits = sorted(circuit.all_qubits())
    qubits_num = len(qubits)
    U1 = cirq.unitary(circuit)
    M = U1 @ np.kron(np.eye(2 ** (qubits_num - 1)), np.array([[1., 0.], [0., 0.]])) @ U1.conj().T
    if p > 1e-5:
        if noise_type == "mixed":
            l = len(noise_list)
            for q in range(qubits_num)[::l]:
                for i in range(l):
                    cur_q = q + i
                    if cur_q < qubits_num:
                        kraus = noise_op_cirq[noise_list[i]](p)(qubits[cur_q])._mixture_()
                        N = 0
                        for E in kraus:
                            F = np.kron(np.eye(2 ** cur_q), np.kron(E[1], np.eye(2 ** (qubits_num - cur_q - 1))))
                            N += E[0] * F @ M @ F.conj().T
                            # kraus_.append(sqrt(E[0]) * E[1])
                        M = N
        elif noise_type == "custom":
            # TODO
            data = load(kraus_file)
            noisy_kraus = data['kraus']
        else:
            # noise = noise_op_cirq[noise_type]
            # noisy_kraus = [cirq.kraus(noise_op_cirq[noise_type](p)(q)) for q in qubits]
            for q in range(qubits_num):
                kraus = noise_op_cirq[noise_type](p)(qubits[q])._mixture_()
                N = 0
                for E in kraus:
                    F = np.kron(np.eye(2 ** q), np.kron(E[1], np.eye(2 ** (qubits_num - q - 1))))
                    N += E[0] * F @ M @ F.conj().T
                    # kraus_.append(sqrt(E[0]) * E[1])
                M = N
    return M


def calculate_lipschitz_based_matrix(circuit_, noise_type_, noise_list_, kraus_file_, noisy_p_):
    print("\n===========The Lipschitz Constant Calculation Start============")
    t_start = time.time()
    M = circuit2matrix(circuit_, noise_type_, noise_list_, kraus_file_, noisy_p_)
    a, _ = np.linalg.eig(M)
    k_base = np.real(max(a) - min(a))
    time_base = time.time() - t_start
    print('Lipschitz K =', k_base)
    print('Elapsed time = %.4fs' % time_base)
    print("============The Lipschitz Constant Calculation End=============")
    return k_base, time_base


def batch_evaluate(model):
    if '.qasm' in model:  # model is qasm_file
        qasm_file = model
        model_name = qasm_file[qasm_file.rfind("/") + 1:-5]
        origin_mq_circuit, origin_cirq_circuit, cirq_qubits = get_origin_circuit(qasm_file)
        # random_mq_circuit, random_cirq_circuit = generating_circuit_with_random_noise(origin_mq_circuit,
        #                                                                               origin_cirq_circuit,
        #                                                                               model_name)
    else:  # case
        model_name = model
        variables, qubits_num = case_params[model_name]
        origin_mq_circuit, origin_cirq_circuit, cirq_qubits = generate_model_circuit(variables, qubits_num, model)
        # random_mq_circuit, random_cirq_circuit = generating_circuit_with_random_noise(origin_mq_circuit,
        #                                                                               origin_cirq_circuit,
        #                                                                               model_name)

    args_iris = [[0.005, 0.003, 0.0001],
                 [0.005, 0.03, 0.0075],
                 [0.0001, 0.005, 0.005],
                 [0.0001, 0.03, 0.005]]
    args_ehc6 = [[0.05, 0.001, 0.0005],
                 [0.075, 0.001, 0.0001],
                 [0.0001, 0.005, 0.003],
                 [0.01, 0.0003, 0.0005]]
    args_ehc8 = [[0.0001, 0.0003, 0.0075],
                 [0.05, 0.001, 0.0075],
                 [0.025, 0.075, 0.0003],
                 [0.0005, 0.005, 0.005]]
    args_fashion8 = [[0.005, 0.075, 0.005],
                     [0.025, 0.03, 0.003],
                     [0.025, 0.005, 0.0003],
                     [0.075, 0.0005, 0.0075]]
    args_aci = [[0.0001, 0.003, 0.0001],
                [0.025, 0.03, 0.0005],
                [0.05, 0.05, 0.001],
                [0.005, 0.005, 0.005]]
    args_fct = [[0.05, 0.075, 0.003],
                [0.05, 0.0003, 0.0001],
                [0.01, 0.01, 0.0075],
                [0.05, 0.075, 0.0075]]
    args_cr = [[0.025, 0.01, 0.0005],
               [0.005, 0.075, 0.005],
               [0.025, 0.0003, 0.0001],
               [0.025, 0.0001, 0.0001]]
    args_qaoa10 = [[0.005, 0.05, 0.0005],
                   [0.0001, 0.01, 0.003],
                   [0.005, 0.075, 0.0075],
                   [0.001, 0.03, 0.0075]]
    args_fashion10 = [[0.005, 0.075, 0.005],
                      [0.025, 0.03, 0.003],
                      [0.025, 0.005, 0.0003],
                      [0.075, 0.0005, 0.0075]]
    args_ehc10 = [[0.075, 0.05, 0.0003],
                  [0.0005, 0.03, 0.001],
                  [0.01, 0.0003, 0.0075],
                  [0.0001, 0.005, 0.001]]
    args_ehc12 = [[0.005, 0.0005, 0.0003],
                  [0.0005, 0.0001, 0.005],
                  [0.075, 0.001, 0.0075],
                  [0.001, 0.01, 0.0001]]
    args_inst16 = [[0.005, 0.0005, 0.0003],
                   [0.0005, 0.0003, 0.005],
                   [0.05, 0.001, 0.0075],
                   [0.001, 0.005, 0.0003]]
    args_map = {
        'iris_4': args_iris,
        'ehc_6': args_ehc6,
        'ehc_8': args_ehc8,
        'fashion_8': args_fashion8,
        'aci_8': args_aci,
        'fct_9': args_fct,
        'cr_9': args_cr,
        'qaoa_10': args_qaoa10,
        'fashion_10': args_fashion10,
        'ehc_10': args_ehc10,
        'ehc_12': args_ehc12,
        'inst_4x4': args_inst16,
    }
    # args = args_map[model_name]
    noise_types = ["bit_flip", "depolarizing", "phase_flip", "mixed"]
    with (open("./results/baseline_comparison.csv", "a+") as csvfile):
        w = csv.writer(csvfile)
        # for i in range(len(args)):
        for i in range(4):
            noise_type = noise_types[i]
            noise_list = ["bit_flip", "depolarizing", "phase_flip"] if noise_type == 'mixed' else []
            kraus_file = None
            if model_name in ['qaoa_20']:
                noise_p = random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075])
                epsilon = random.choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075])
                delta = random.choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.0075])
            else:
                args = args_map[model_name]
                noise_p, epsilon, delta = args[i][0], args[i][1], args[i][2]
            origin_mq_circuit_ = mindquantum.Circuit(origin_mq_circuit)
            origin_cirq_circuit_ = cirq.Circuit(origin_cirq_circuit)
            final_mq_circuit, final_cirq_circuit, _ = generating_circuit_with_specified_noise(
                origin_mq_circuit_, origin_cirq_circuit_, noise_type, noise_list, kraus_file, noise_p, model_name)

            noise_ = noise_type.replace('_', ' ')
            # Tensor-based
            try:
                with time_limit():
                    final_k, final_time, final_bias_kernel = calculate_lipschitz(final_cirq_circuit, cirq_qubits)
                    res = 'YES' if verification(final_k, epsilon, delta) else 'NO'
                    if model_name in ['inst_4x4', 'qaoa_20']:
                        w.writerow([model_name, noise_, noise_p, (epsilon, delta),
                                    '-', 'TO', '%.5f' % final_k, '%.2f' % final_time, res])
                        continue
                    else:
                        w.writerow([model_name, noise_, noise_p, (epsilon, delta),
                                    '%.5f' % k_base, '%.2f' % time_base, '%.5f' % final_k, '%.2f' % final_time, res])
            except TimeoutException:
                print('Time out!')
                w.writerow([model_name, noise_, noise_p, (epsilon, delta),
                            '-', 'TO', '-', 'TO', '-'])
            except Exception:
                w.writerow([model_name, noise_, noise_p, (epsilon, delta),
                            '-', 'OOM', '-', 'OOM', '-'])
                raise

            # Matrix-based
            try:
                with time_limit():
                    k_base, time_base = calculate_lipschitz_based_matrix(origin_cirq_circuit, noise_type, noise_list,
                                                                         kraus_file, noise_p)
                    w.writerow([model_name, noise_, noise_p, (epsilon, delta),
                                '%.5f' % k_base, '%.2f' % time_base, '%.5f' % final_k, '%.2f' % final_time, res])
            except TimeoutException:
                print('Time out!')
                w.writerow([model_name, noise_, noise_p, (epsilon, delta),
                            '-', 'TO', '%.5f' % final_k, '%.2f' % final_time, res])
            except Exception:
                w.writerow([model_name, noise_, noise_p, (epsilon, delta),
                            '-', 'OOM', '-', 'OOM', '-'])
                raise


batch_evaluate('./qasm_models/iris_4.qasm')
# batch_evaluate('./qasm_models/HFVQE/ehc_6.qasm')
# batch_evaluate('./qasm_models/HFVQE/ehc_8.qasm')
# batch_evaluate('./qasm_models/fashion_8.qasm')
# batch_evaluate('aci_8')
# batch_evaluate('fct_9')
# batch_evaluate('cr_9')
# batch_evaluate('./qasm_models/QAOA/qaoa_10.qasm')
# batch_evaluate('./qasm_models/HFVQE/ehc_10.qasm')
# batch_evaluate('./qasm_models/HFVQE/ehc_12.qasm')
# batch_evaluate('./qasm_models/inst/inst_4x4.qasm')
# batch_evaluate('./qasm_models/QAOA/qaoa_20.qasm')
