from common_interface import *
from sys import argv
import os
import csv

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


if str(argv[1]) != "verify":
    # python global_verif.py ./qasm_models/HFVQE/ehc_6.qasm phase_flip 0.0001
    arg_num = len(argv)
    noise_list = []
    kraus_file = ''
    if arg_num <= 2:  # random noise
        noise_type = random.choice(noise_ops)
        noise = noise_op_mq[noise_type].__name__
        noise = noise[0: noise.index("Channel")]
        noise_p = float(round(random.uniform(0, 0.2), 5))  # 随机数的精度round(数值，精度)
        # file_name = "{}_{}_{}".format(model_name, noise, str(noise_p))
    else:
        noise_type = str(argv[2])
        noise_p = float(argv[arg_num - 1])
        if noise_type == 'mixed':
            noise_list = [i for i in argv[3: arg_num - 1]]
        elif noise_type == 'custom':
            kraus_file = argv[3]

    if '.qasm' in argv[1]:  # qasm_file
        qasm_file = str(argv[1])
        model_name = qasm_file[qasm_file.rfind("/") + 1:-5]
        origin_mq_circuit, origin_cirq_circuit, cirq_qubits = get_origin_circuit(qasm_file)
        origin_mq_circuit_ = mindquantum.Circuit(origin_mq_circuit)
        origin_cirq_circuit_ = cirq.Circuit(origin_cirq_circuit)
        random_mq_circuit, random_cirq_circuit = generating_circuit_with_random_noise(origin_mq_circuit_,
                                                                                      origin_cirq_circuit_, model_name)
    else:  # case
        model_name = str(argv[1])
        variables, qubits_num = case_params[model_name]
        origin_mq_circuit, origin_cirq_circuit, cirq_qubits = generate_model_circuit(variables, qubits_num, model_name)
        origin_mq_circuit_ = mindquantum.Circuit(origin_mq_circuit)
        origin_cirq_circuit_ = cirq.Circuit(origin_cirq_circuit)
        random_mq_circuit, random_cirq_circuit = generating_circuit_with_random_noise(origin_mq_circuit_,
                                                                                      origin_cirq_circuit_, model_name)

    random_mq_circuit_ = mindquantum.Circuit(random_mq_circuit)
    random_cirq_circuit_ = cirq.Circuit(random_cirq_circuit)
    final_mq_circuit, final_cirq_circuit = generating_circuit_with_specified_noise(
        random_mq_circuit_, random_cirq_circuit_, noise_type, noise_list, kraus_file, noise_p, model_name)

    # origin_k, origin_time, origin_bias_kernel = calculate_lipschitz(origin_cirq_circuit, cirq_qubits)
    random_k, random_time, random_bias_kernel = calculate_lipschitz(random_cirq_circuit, cirq_qubits)
    final_k, final_time, final_bias_kernel = calculate_lipschitz(final_cirq_circuit, cirq_qubits)

    with (open("./results/global_results.csv", "a+") as csvfile):
        w = csv.writer(csvfile)

        epsilon = random.choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075])
        delta = random.choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.0075])

        # start = time.time()
        # res = 'YES' if verification(origin_k, epsilon, delta) else 'NO'
        # origin_time += time.time() - start

        start = time.time()
        res = 'YES' if verification(random_k, epsilon, delta) else 'NO'
        random_time += time.time() - start

        # start = time.time()
        # res = 'YES' if verification(final_k, epsilon, delta) else 'NO'
        # final_time += time.time() - start

        noise_ = noise_type.replace('_', ' ')
        # Tensor-based
        # c0
        try:
            with time_limit():
                origin_k, origin_time, origin_bias_kernel = calculate_lipschitz(origin_cirq_circuit, cirq_qubits)
                start = time.time()
                res = 'YES' if verification(origin_k, epsilon, delta) else 'NO'
                origin_time += time.time() - start
                w.writerow([model_name, 'c_0', '-', '-', (epsilon, delta),
                            '%.5f' % origin_k, '%.2f' % origin_time, res])
        except TimeoutException:
            print('Time out!')
            w.writerow([model_name, 'c_0', '-', '-', (epsilon, delta),
                        '-', 'TO', '-'])
        except Exception:
            if model_name in ['inst_4x4', 'qaoa_20']:
                w.writerow([model_name, 'c_0', '-', '-', (epsilon, delta),
                            '-', 'OOM', '-'])
            raise

        # c1
        try:
            with time_limit():
                random_k, random_time, random_bias_kernel = calculate_lipschitz(random_cirq_circuit, cirq_qubits)
                start = time.time()
                res = 'YES' if verification(random_k, epsilon, delta) else 'NO'
                random_time += time.time() - start
                w.writerow([model_name, 'c_1', '-', '-', (epsilon, delta),
                            '%.5f' % random_k, '%.2f' % random_time, res])
        except TimeoutException:
            print('Time out!')
            w.writerow([model_name, 'c_1', '-', '-', (epsilon, delta),
                        '-', 'TO', '-'])
        except Exception:
            if model_name in ['inst_4x4', 'qaoa_20']:
                w.writerow([model_name, 'c_1', '-', '-', (epsilon, delta),
                            '-', 'OOM', '-'])
            raise

        # c2
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
            w.writerow([model_name, 'c_2', noise_, noise_p, (epsilon, delta),
                        '-', 'OOM', '-'])
            raise
else:
    # python qlipschitz.py verify k epsilon delta
    k = float(argv[2])
    epsilon = float(argv[3])
    delta = float(argv[4])
    # flag, k, bias_kernel, total_time = verification(epsilon, delta)
    flag = verification(k, epsilon, delta)
