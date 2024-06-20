from common_interface import *
from sys import argv
import os
import csv

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

arg_num = len(argv)
if str(argv[1]) != "verify":
    # python global_verif.py ./qasm_models/HFVQE/ehc_6.qasm phase_flip 0.0001
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
            noise_list_ = [noise_op_mq[i].__name__ for i in noise_list]
            noise_list_ = [i[0: i.index("Channel")] for i in noise_list_]
            noise_ = "mixed_{}_{}".format('_'.join(noise_list_), noise_p)
        elif noise_type == 'custom':
            kraus_file = argv[3]
            noise_ = "custom_{}_{}".format(kraus_file[kraus_file.rfind('/') + 1:-4], noise_p)
        else:
            noise_ = noise_op_mq[noise_type].__name__
            noise_ = noise_[0: noise_.index("Channel")]
            noise_ = "{}_{}".format(noise_, noise_p)

    if '.qasm' in argv[1]:  # qasm_file
        qasm_file = str(argv[1])
        model_name = qasm_file[qasm_file.rfind("/") + 1:-5]
        filedir = "{}_{}".format(model_name, noise_)
        origin_mq_circuit, origin_cirq_circuit, cirq_qubits = get_origin_circuit(qasm_file, True, filedir)
    else:  # case
        model_name = str(argv[1])
        variables, qubits_num = case_params[model_name]
        filedir = "{}_{}".format(model_name, noise_)
        origin_mq_circuit, origin_cirq_circuit, cirq_qubits = generate_model_circuit(variables, qubits_num,
                                                                                     model_name, True, filedir)

    origin_mq_circuit_ = mindquantum.Circuit(origin_mq_circuit)
    origin_cirq_circuit_ = cirq.Circuit(origin_cirq_circuit)
    random_mq_circuit, random_cirq_circuit = generating_circuit_with_random_noise(origin_mq_circuit_,
                                                                                  origin_cirq_circuit_,
                                                                                  model_name, True, filedir)

    random_mq_circuit_ = mindquantum.Circuit(random_mq_circuit)
    random_cirq_circuit_ = cirq.Circuit(random_cirq_circuit)
    final_mq_circuit, final_cirq_circuit, _ = generating_circuit_with_specified_noise(
        random_mq_circuit_, random_cirq_circuit_, noise_type, noise_list, kraus_file, noise_p, model_name, True,
        filedir)

    # with (open("./results/{}/{}_{}.csv".format(model_name, model_name, noise_), "a+") as csvfile):
    #     w = csv.writer(csvfile)
    # c0: noiseless
    try:
        with time_limit():
            origin_k, origin_time, origin_bias_kernel = calculate_lipschitz(origin_cirq_circuit, cirq_qubits)
    except TimeoutException:
        print('Time Out!')
    except Exception:
        if model_name in OOM_model_list:
            print('Out of Memory')
        raise

    # c1: random noise
    try:
        with time_limit():
            random_k, random_time, random_bias_kernel = calculate_lipschitz(random_cirq_circuit, cirq_qubits)
    except TimeoutException:
        print('Time Out!')
    except Exception:
        if model_name in OOM_model_list:
            print('Out of Memory')
        raise

    # c2: random & specified noise
    # circuit_type = 'random & {}_{}'.format(noise_type.replace('_', '-'), noise_p)
    try:
        with time_limit():
            final_k, final_time, final_bias_kernel = calculate_lipschitz(final_cirq_circuit, cirq_qubits)
    except TimeoutException:
        print('Time Out!')
    except Exception:
        if model_name in OOM_model_list:
            print('Out of Memory')
        raise
else:
    # python global_verif.py verify k epsilon delta
    # or
    # python global_verif.py verify k0 k1 k2 epsilon delta
    if arg_num == 5:
        k = float(argv[2])
        epsilon = float(argv[3])
        delta = float(argv[4])
        # flag, k, bias_kernel, total_time = verification(epsilon, delta)
        flag = verification(k, epsilon, delta)
    else:  # > 5
        epsilon = float(argv[arg_num-2])
        delta = float(argv[arg_num-1])
        for k in argv[2: arg_num-2]:
            k = float(k)
            flag = verification(k, epsilon, delta)
