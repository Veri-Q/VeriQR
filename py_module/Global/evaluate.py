import os
import csv
import matplotlib.pyplot as plt
from common_interface import *

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def batch_evaluate(model):
    if '.qasm' in model:  # model is qasm_file
        qasm_file = model
        model_name = qasm_file[qasm_file.rfind("/") + 1:-5]
        origin_mq_circuit, origin_cirq_circuit, cirq_qubits = get_origin_circuit(qasm_file)
        origin_mq_circuit_ = mindquantum.Circuit(origin_mq_circuit)
        origin_cirq_circuit_ = cirq.Circuit(origin_cirq_circuit)
        random_mq_circuit, random_cirq_circuit = generating_circuit_with_random_noise(origin_mq_circuit_,
                                                                                      origin_cirq_circuit_, model_name)
    else:  # case
        model_name = model
        variables, qubits_num = case_params[model_name]
        origin_mq_circuit, origin_cirq_circuit, cirq_qubits = generate_model_circuit(variables, qubits_num, model)
        origin_mq_circuit_ = mindquantum.Circuit(origin_mq_circuit)
        origin_cirq_circuit_ = cirq.Circuit(origin_cirq_circuit)
        random_mq_circuit, random_cirq_circuit = generating_circuit_with_random_noise(origin_mq_circuit_,
                                                                                      origin_cirq_circuit_, model_name)
    # no noise
    origin_k, origin_time, origin_bias_kernel = calculate_lipschitz(origin_cirq_circuit, cirq_qubits)
    # random noise
    random_k, random_time, random_bias_kernel = calculate_lipschitz(random_cirq_circuit, cirq_qubits)

    epsilons = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075]
    deltas = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.0075]
    probs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
    noise_types = ["bit_flip", "depolarizing", "phase_flip", "mixed"]
    kraus_file = None
    with (open("./results/global_results.csv", "a+") as csvfile):
        w = csv.writer(csvfile)
        epsilon = random.choice(epsilons)
        delta = random.choice(deltas)
        # origin_time_ = origin_time
        start = time.time()
        res = 'YES' if verification(origin_k, epsilon, delta) else 'NO'
        origin_time += time.time() - start
        w.writerow([model_name, 'c_0', '-', '-', (epsilon, delta),
                    '%.5f' % origin_k, '%.2f' % origin_time, res])

        # random_time_ = random_time
        start = time.time()
        res = 'YES' if verification(random_k, epsilon, delta) else 'NO'
        random_time += time.time() - start
        w.writerow([model_name, 'c_1', '-', '-', (epsilon, delta),
                    '%.5f' % random_k, '%.2f' % random_time, res])

        for noise_type in noise_types:
            # for noise_type in ["phase_flip"]:
            noise_list = noise_types[:3] if noise_type == 'mixed' else []
            for _ in range(3):
                noise_p = random.choice(probs)
                random_mq_circuit_ = mindquantum.Circuit(random_mq_circuit)
                random_cirq_circuit_ = cirq.Circuit(random_cirq_circuit)
                final_mq_circuit, final_cirq_circuit = generating_circuit_with_specified_noise(
                    random_mq_circuit_, random_cirq_circuit_, noise_type, noise_list, kraus_file, noise_p, model_name)

                noise_ = noise_type.replace('_', ' ')
                # Tensor-based
                try:
                    with time_limit():
                        final_k, final_time, final_bias_kernel = calculate_lipschitz(final_cirq_circuit, cirq_qubits)
                        start = time.time()
                        res = 'YES' if verification(final_k, epsilon, delta) else 'NO'
                        final_time += time.time() - start
                        w.writerow([model_name, 'c_2', noise_, noise_p, (epsilon, delta),
                                    '%.5f' % final_k, '%.2f' % final_time, res])
                except TimeoutException:
                    print('Time out!')
                    w.writerow([model_name, 'c_2', noise_, noise_p, (epsilon, delta),
                                '-', 'TO', '-'])
                except Exception:
                    if model_name in ['inst_4x4', 'qaoa_20']:
                        w.writerow([model_name, 'c_2', noise_, noise_p, (epsilon, delta),
                                    '-', 'OOM', '-'])
                    raise


def batch_evaluate_for_plot(model):
    if '.qasm' in model:  # model is qasm_file
        qasm_file = model
        model_name = qasm_file[qasm_file.rfind("/") + 1:-5]
        origin_mq_circuit, origin_cirq_circuit, cirq_qubits = get_origin_circuit(qasm_file)
        random_mq_circuit, random_cirq_circuit = generating_circuit_with_random_noise(origin_mq_circuit,
                                                                                      origin_cirq_circuit, model_name)
    else:  # case
        model_name = model
        variables, qubits_num = case_params[model_name]
        origin_mq_circuit, origin_cirq_circuit, cirq_qubits = generate_model_circuit(variables, qubits_num, model)
        origin_mq_circuit_ = mindquantum.Circuit(origin_mq_circuit)
        origin_cirq_circuit_ = cirq.Circuit(origin_cirq_circuit)
        random_mq_circuit, random_cirq_circuit = generating_circuit_with_random_noise(origin_mq_circuit_,
                                                                                      origin_cirq_circuit_, model_name)
    probs = [0.005 * i for i in range(1, 40 + 1)]
    x = np.arange(0, 0.201, 0.005)
    with (open("./results/global_results_v2.csv", "a+") as csvfile):
        w = csv.writer(csvfile)
        for _ in range(1):
            epsilon = random.choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075])
            delta = random.choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.0075])
            # no noise
            origin_k, origin_time, origin_bias_kernel = calculate_lipschitz(origin_cirq_circuit, cirq_qubits)
            start = time.time()
            res = 'YES' if verification(origin_k, epsilon, delta) else 'NO'
            origin_time += time.time() - start
            w.writerow([model_name, 'c_0', (epsilon, delta), 'None', 'None',
                        '%.5f' % origin_k, '%.2f' % origin_time, res])
            # random noise
            random_k, random_time, random_bias_kernel = calculate_lipschitz(random_cirq_circuit, cirq_qubits)
            start = time.time()
            res = 'YES' if verification(random_k, epsilon, delta) else 'NO'
            random_time += time.time() - start
            w.writerow([model_name, 'c_1', (epsilon, delta), 'None', 'None',
                        '%.5f' % random_k, '%.2f' % random_time, res])

            y_bf, y_dp, y_pf, y_mixed = [origin_k], [origin_k], [origin_k], [origin_k]
            for noise_type in ["bit_flip", "depolarizing", "phase_flip", "mixed"]:
                # for noise_type in ["phase_flip"]:
                noise_list = ["bit_flip", "depolarizing", "phase_flip"] if noise_type == 'mixed' else []
                kraus_file = None
                for noisy_p in probs:
                    random_mq_circuit_ = mindquantum.Circuit(random_mq_circuit)
                    random_cirq_circuit_ = cirq.Circuit(random_cirq_circuit)
                    final_mq_circuit, final_cirq_circuit = generating_circuit_with_specified_noise(
                        random_mq_circuit_, random_cirq_circuit_, noise_type, noise_list, kraus_file, noisy_p,
                        model_name)
                    # specified noise
                    final_k, final_time, final_bias_kernel = calculate_lipschitz(final_cirq_circuit, cirq_qubits)
                    start = time.time()
                    res = 'YES' if verification(final_k, epsilon, delta) else 'NO'
                    final_time += time.time() - start
                    w.writerow([model_name, 'c_2', (epsilon, delta), noise_type.replace('_', ' '), '%.3f' % noisy_p,
                                '%.5f' % final_k, '%.2f' % final_time, res])

                    if noise_type == "bit_flip":
                        y_bf.append(final_k)
                    elif noise_type == "depolarizing":
                        y_dp.append(final_k)
                    elif noise_type == "phase_flip":
                        y_pf.append(final_k)
                    elif noise_type == "mixed":
                        y_mixed.append(final_k)
            lines = plt.plot(x, y_bf, x, y_dp, x, y_pf, x, y_mixed, 'o')
            plt.setp(lines[0])
            plt.setp(lines[1])
            plt.setp(lines[2], linestyle='-')
            plt.setp(lines[3], linestyle='-', marker='^', markersize=1)
            # plt.xlim((0, 0.2))
            # plt.ylim((0, 1))
            # plt.xticks(np.arange(0, 0.2, 0.02))
            # plt.yticks(np.arange(0, 1, 0.1))
            plt.xlabel('p')
            plt.ylabel('K*')
            plt.legend(('${bit flip}$ noise', '${depolarizing}$ noise',
                        '${phase flip}$ noise', '${mixed}$ noise'), loc='upper right')
            plt.title('The Lipschitz constant K* of the ${}$ model'.format(model_name))
            plt.show()
            plt.savefig('./results/figures/{}.pdf'.format(model_name))


# file = str(sys.argv[1])
# batch_evaluate(file)

# batch_evaluate_for_plot('./qasm_models/iris.qasm')
# gc.collect()
# batch_evaluate_for_plot('./qasm_models/HFVQE/ehc_6.qasm')
# gc.collect()
# batch_evaluate_for_plot('./qasm_models/HFVQE/ehc_8.qasm')
# gc.collect()
# batch_evaluate_for_plot('./qasm_models/fashion_8.qasm')
# gc.collect()
# batch_evaluate_for_plot('aci_8')
# gc.collect()
# batch_evaluate_for_plot('fct_9')
# gc.collect()
# batch_evaluate_for_plot('cr_9')
# gc.collect()
# batch_evaluate_for_plot('./qasm_models/QAOA/qaoa_10.qasm')
# gc.collect()
# batch_evaluate_for_plot('./qasm_models/HFVQE/ehc_10.qasm')
# gc.collect()
# batch_evaluate_for_plot('./qasm_models/HFVQE/ehc_12.qasm')
# gc.collect()
# batch_evaluate_for_plot('./qasm_models/inst/inst_4x4.qasm')
# gc.collect()
# batch_evaluate_for_plot('./qasm_models/QAOA/qaoa_20.qasm')
# gc.collect()

batch_evaluate('./qasm_models/iris_4.qasm')
gc.collect()
batch_evaluate('./qasm_models/HFVQE/ehc_6.qasm')
gc.collect()
batch_evaluate('./qasm_models/HFVQE/ehc_8.qasm')
gc.collect()
batch_evaluate('./qasm_models/fashion_8.qasm')
gc.collect()
batch_evaluate('aci_8')
gc.collect()
batch_evaluate('fct_9')
gc.collect()
batch_evaluate('cr_9')
gc.collect()
batch_evaluate('./qasm_models/QAOA/qaoa_10.qasm')
gc.collect()
batch_evaluate('./qasm_models/HFVQE/ehc_10.qasm')
gc.collect()
batch_evaluate('./qasm_models/HFVQE/ehc_12.qasm')
gc.collect()
batch_evaluate('./qasm_models/inst/inst_4x4.qasm')
gc.collect()
batch_evaluate('./qasm_models/QAOA/qaoa_20.qasm')
gc.collect()


def plot_results_figure(model_name):
    with open("./results/global_results_v2.csv") as f:
        if model_name in ['aci_8', 'fct_9', 'cr_9']:
            x = np.arange(0.002, 0.201, 0.002)
        else:
            x = np.arange(0.001, 0.201, 0.001)  # for others
        y_bf, y_dp, y_pf, y_mixed = [], [], [], []
        origin_k = 1.0
        random_k = 1.0
        for row in csv.reader(f, skipinitialspace=True):
            if row == [] or row[0] != model_name:
                continue
            if row[1] == 'c_0':
                origin_k = float(row[5])
                continue

            if row[1] == 'c_1':
                random_k = float(row[5])
                # y_bf, y_dp, y_pf, y_mixed = [k], [k], [k], [k]
                continue

            noise_type = row[3]
            k = float(row[5])
            if noise_type == "bit flip":
                y_bf.append(k)
            elif noise_type == "depolarizing":
                y_dp.append(k)
            elif noise_type == "phase flip":
                y_pf.append(k)
            elif noise_type == "mixed":
                y_mixed.append(k)

        plt.scatter(0.0, origin_k, label='origin', marker='o')
        plt.scatter(0.0, random_k, label='random noise', marker='*')
        lines = plt.plot(x, y_bf, x, y_dp, x, y_pf, x, y_mixed, 'o')
        plt.setp(lines[0])
        plt.setp(lines[1])
        plt.setp(lines[2], linestyle='-')
        plt.setp(lines[3], linestyle='-', markersize=1)
        # lines = plt.plot([0.0], [origin_k], [0.0], [random_k], 
        #                  x, y_bf, x, y_dp, x, y_pf, x, y_mixed, 'o')
        # plt.setp(lines[0], linestyle='-', marker='o', markersize=6)
        # plt.setp(lines[1], linestyle='-', marker='*', markersize=6)
        # plt.setp(lines[2])
        # plt.setp(lines[3])
        # plt.setp(lines[4], linestyle='-')
        # plt.setp(lines[5], linestyle='-', markersize=1)
        plt.xlabel('p')
        plt.ylabel('K*')
        # plt.xlim((0, 0.2))
        # plt.ylim((0, 1))
        # plt.xticks(np.arange(0, 0.2, 0.02))
        # plt.yticks(np.arange(0, 1, 0.1))

        plt.legend(('origin', 'random noise', '${bit}$-${flip}$ noise', '${depolarizing}$ noise',
                    '${phase}$-${flip}$ noise', '${mixed}$ noise'), loc='upper right')
        if '_' not in model_name or model_name.index('_') + 1 == len(model_name) - 1:
            model_ = "${}$".format(model_name)
        else:
            model_ = "${}_{}$".format(model_name[0:model_name.index('_')],
                                      '{' + model_name[model_name.index('_') + 1:len(model_name)] + '}')
            # plt.title('The Lipschitz constant K* of the ${}_{}$ model'.format(
            #                     model_name[0:model_name.index('_')], model_name[model_name.index('_')+1:]))
        plt.title('The {} model'.format(model_))
        plt.show()
        plt.savefig('./results/figures/{}.pdf'.format(model_name))


model_names = ['iris_4', 'ehc_6', 'ehc_8', 'fashion_8', 'aci_8', 'fct_9', 'cr_9', 'qaoa_10', 'ehc_10', 'ehc_12']
# plot_results_figure('iris_4')
