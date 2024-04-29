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

    # M = U1.conj().T @ M @ U1
    return M


model_name = str(sys.argv[1])
if '.qasm' in model_name:
    model_name = model_name[model_name.rfind('/') + 1:-5]
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
    # w.writerow([model_name, noise_type.replace('_', ' '), noisy_p, (epsilon, delta),
    #             '%.5f' % k, res, '%.2f' % total_time])
    w.writerow([model_name, noise_type.replace('_', ' '), noisy_p, (epsilon, delta),
                res, '%.5f' % k_base, '%.2f' % time_base, '%.5f' % k_tn, '%.2f' % time_tn])

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
