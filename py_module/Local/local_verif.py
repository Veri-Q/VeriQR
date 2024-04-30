from VeriL import RobustnessVerifier, PureRobustnessVerifier
from common_interface import *
from prettytable import PrettyTable
from sys import argv

digits = '36'
GET_NEW_DATASET = False
ADVERSARY_EXAMPLE = False
noise_p = 0

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
    origin_dataset_size = label.shape[0]

    type = 'npz'
    model_name = data_file[data_file.rfind('/') + 1: data_file.rfind('_')]
    file_name = '{}_{}_{}_{}.csv'.format(model_name, eps, n, state_flag)  # 默认文件名
else:
    # '.qasm' in str(argv[1])
    # for example:
    # python batch_check.py ./model_and_data/mnist56.qasm ./model_and_data/mnist56_data.npz 0.001 1 pure true (argv[6])
    # python batch_check.py ./model_and_data/mnist56.qasm ./model_and_data/mnist56_data.npz 0.001 1 pure true phase_flip 0.001 (argv[8])
    # python batch_check.py ./model_and_data/FashionMNIST.qasm ./model_and_data/FashionMNIST_data.npz 0.001 1 pure (argv[5])
    # python batch_check.py ./model_and_data/FashionMNIST.qasm ./model_and_data/FashionMNIST_data.npz 0.001 1 pure phase_flip 0.001 (argv[7])
    # python batch_check.py ./model_and_data/iris.qasm ./model_and_data/iris_data.npz 0.001 1 mixed (argv[5])
    # python batch_check.py ./model_and_data/iris.qasm ./model_and_data/iris_data.npz 0.001 1 mixed phase_flip 0.001 (argv[7])
    qasm_file = str(argv[1])
    data_file = str(argv[2])
    eps = float(argv[3])
    n = int(argv[4])
    state_flag = str(argv[5])
    model_name = qasm_file[qasm_file.rfind('/') + 1:-5]

    if 'mnist' in data_file:  # digits != '36'
        ADVERSARY_EXAMPLE = (str(argv[6]) == 'true')
        if '_data' in data_file:  # digits != '36'
            digits = data_file[data_file.rfind('_data') - 2: data_file.rfind('_data')]

    noise_list = []
    kraus_file = None
    arg_num = len(argv)
    if arg_num > 7:
        noise_type = argv[7]
        noise_p = float(argv[arg_num - 1])
        if noise_type == 'mixed':
            noise_list = [i for i in argv[8: arg_num - 1]]
            print("noise_list: ", noise_list)
        elif noise_type == 'custom':
            kraus_file = argv[8]
    # else:
    #     noise_type = random.choice(noise_ops)
    #     noise_p = float(round(random.uniform(0, 0.2), 5))  # 随机数的精度round(数值，精度)
    circuit = qasm2mq(qasm_file)
    circuit, kraus = generating_circuit_with_random_noise(circuit, model_name)
    kraus, noise_name = generating_circuit_with_specified_noise(circuit, kraus, noise_type, noise_list,
                                                                kraus_file, noise_p, model_name)

    DATA = load(data_file)
    O = DATA['O']
    data = DATA['data']
    label = DATA['label']
    origin_dataset_size = label.shape[0]

    type = 'qasm'
    file_name = '{}_{}_{}_{}_{}_{}.csv'.format(model_name, eps, n, state_flag, noise_p, noise_name)  # 默认文件名
    # file_name = '{}_{}_{}_{}.csv'.format(
    # qasm_file[qasm_file.rfind('/') + 1:-5], eps, n, state_flag)  # 默认文件名

verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

ac = PrettyTable()
time = PrettyTable()
ac.add_column('epsilon', ['Robust Bound', 'Robustness Algorithm'])
time.add_column('epsilon', ['Robust Bound', 'Robustness Algorithm'])
for j in range(n):
    c_eps = eps * (j + 1)
    if 'mnist' in data_file:
        ac_temp, time_temp, new_data, new_labels = verifier(kraus, O, data, label, c_eps, type,
                                                            GET_NEW_DATASET, 0, ADVERSARY_EXAMPLE,
                                                            digits, 'mnist')
    else:
        ac_temp, time_temp, new_data, new_labels = verifier(kraus, O, data, label, c_eps, type,
                                                            GET_NEW_DATASET, origin_dataset_size)

    ac.add_column('{:e}'.format(c_eps),
                  ['{:.2f}'.format(ac_temp[0] * 100), '{:.2f}'.format(ac_temp[1] * 100)])
    time.add_column('{:e}'.format(c_eps),
                    ['{:.4f}'.format(time_temp[0]), '{:.4f}'.format(time_temp[1])])
    # np.savez('./model_and_data/adversary_training/{}_{}_data.npz'.format(model_name, c_eps),
    #          O=O, data=new_data, label=new_labels, kraus=kraus)

file_path = './results/result_tables/' + file_name

with open(file_path, 'w', newline='') as f_output:
    f_output.write(ac.get_csv_string())
    f_output.write('\n')
    f_output.write(time.get_csv_string())
    f_output.close()
    print(file_name + " saved successfully! ")
