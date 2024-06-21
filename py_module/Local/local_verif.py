from VeriL import RobustnessVerifier, PureRobustnessVerifier
from common_interface import *
from prettytable import PrettyTable
from sys import argv

Save_Figure = True

epsilon = 0.
batch_num = 1
state_type = 'mixed'
digits = '36'
GET_NEW_DATASET = False
ADVERSARY_EXAMPLE = False
type = 'qasm'
origin_dataset_size = 0
data_file, kraus, O, data, label = None, None, None, None, None
result_file_name = ''

if '.npz' in str(argv[1]):
    # for example:
    # python local_verif.py qubit.npz 0.001 1 mixed true
    data_file = str(argv[1])
    epsilon = float(argv[2])
    batch_num = int(argv[3])
    state_type = str(argv[4])
    GET_NEW_DATASET = (str(argv[5]) == 'true')

    model_name = data_file[data_file.rfind('/') + 1: -4]
    DATA = load(data_file)
    kraus = DATA['kraus']
    O = DATA['O']
    data = DATA['data']
    label = DATA['label']
    origin_dataset_size = label.shape[0]
    type = 'npz'

    if model_name == 'qubit':
        n_qubits = 1
        p = float(round(random.uniform(0, 0.2), 5))
        E = noise_op_map[random.choice(noise_ops)](p).matrix()
        kraus_ = []
        for e in E:
            kraus_.append(e @ kraus[0])
        random_kraus = np.array(kraus_)
        print('random kraus.shape', random_kraus.shape)
        np.savez('./model_and_data/qubit_random.npz', O=O, data=data, label=label, kraus=random_kraus)

        arg_num = len(argv)
        if arg_num > 6:
            noise_p = float(argv[arg_num - 1])
            noise_type = argv[6]
            if noise_type == 'mixed':
                # noise_list = [i for i in argv[6: arg_num - 1]]
                # E = noise_op_map[random.choice(noise_ops)](noise_p).matrix()
                noise_op = noise_op_map[random.choice(noise_ops)]
                noise_name = noise_op.__name__
                noise_name = noise_name[0: noise_name.index("Channel")]
                E = noise_op(noise_p).matrix()
            elif noise_type == 'custom':
                kraus_file = argv[6]
                custom_kraus = load(kraus_file)['kraus']
                dim = 2 ** n_qubits
                for i in range(custom_kraus.shape[0]):
                    if custom_kraus[i].shape[0] != dim or custom_kraus[i].shape[1] != dim:
                        raise RuntimeError("The dimension of the kraus operator is {}, not consistent with "
                                           "the circuit's ({}, {})! ".format(custom_kraus[i].shape, dim, dim))
                E = custom_kraus[0]
                noise_name = "custom_{}".format(kraus_file[kraus_file.rfind('/') + 1:-4])
            else:
                # E = noise_op_map[noise_type](noise_p).matrix()
                noise_op = noise_op_map[noise_type]
                noise_name = noise_op.__name__
                noise_name = noise_name[0: noise_name.index("Channel")]
                E = noise_op(noise_p).matrix()
            # print(E)
            new_kraus = []
            for k in random_kraus:
                for e in E:
                    new_kraus.append(e @ k)
            final_kraus = np.array(new_kraus)
            print('new kraus.shape', final_kraus.shape)
            np.savez('./model_and_data/qubit_{}_{}.npz'.format(noise_type, noise_p),
                     O=O, data=data, label=label, kraus=final_kraus)
            result_file_name = '{}_{}×{}_{}_{}_{}.csv'.format(model_name, epsilon, batch_num, state_type,
                                                              noise_name, noise_p)
    else:
        # default filename
        result_file_name = '{}_{}×{}_{}.csv'.format(model_name, epsilon, batch_num, state_type)
elif '.qasm' in str(argv[1]):
    # for example:
    # python local_verif.py ./model_and_data/mnist01.qasm ./model_and_data/mnist01_data.npz 0.001 1 pure true true (argv[7])
    # python local_verif.py ./model_and_data/mnist56.qasm ./model_and_data/mnist56_data.npz 0.001 1 pure true true phase_flip 0.001
    # python local_verif.py ./model_and_data/fashion8.qasm ./model_and_data/fashion8_data.npz 0.001 1 pure false true (argv[7])
    # python local_verif.py ./model_and_data/fashion8.qasm ./model_and_data/fashion8_data.npz 0.001 1 pure false true phase_flip 0.001
    # python local_verif.py ./model_and_data/iris.qasm ./model_and_data/iris_data.npz 0.001 1 mixed false true (argv[7])
    # python local_verif.py ./model_and_data/iris.qasm ./model_and_data/iris_data.npz 0.001 1 mixed false true phase_flip 0.001
    qasm_file = str(argv[1])
    data_file = str(argv[2])
    epsilon = float(argv[3])
    batch_num = int(argv[4])
    state_type = str(argv[5])

    model_name = qasm_file[qasm_file.rfind('/') + 1:-5]
    DATA = load(data_file)
    O = DATA['O']
    data = DATA['data']
    label = DATA['label']
    origin_dataset_size = label.shape[0]
    type = 'qasm'

    if 'mnist' in data_file:
        ADVERSARY_EXAMPLE = (str(argv[6]) == 'true')
        digits = model_name[5:]
    else:
        ADVERSARY_EXAMPLE = False

    GET_NEW_DATASET = (str(argv[7]) == 'true')

    # add random noise
    circuit, kraus = qasm2mq_with_kraus(qasm_file, True)
    circuit, random_kraus = generating_circuit_with_random_noise(circuit, model_name, True)

    # specified noise
    noise_list = []
    kraus_file = None
    arg_num = len(argv)
    if arg_num > 8:
        noise_p = float(argv[arg_num - 1])
        noise_type = argv[8]
        if noise_type == 'mixed':
            noise_list = [i for i in argv[9: arg_num - 1]]
            print("noise_list: ", noise_list)
        elif noise_type == 'custom':
            kraus_file = argv[9]

        final_kraus, noise_name = generating_circuit_with_specified_noise(circuit, random_kraus, noise_type, noise_list,
                                                                          kraus_file, noise_p, model_name, True)
        # noise_name = noise_name_map[noise_type]
        # default filename
        result_file_name = '{}_{}×{}_{}_{}_{}.csv'.format(model_name, epsilon, batch_num, state_type,
                                                          noise_name, noise_p)
    else:
        result_file_name = '{}_{}×{}_{}.csv'.format(model_name, epsilon, batch_num, state_type)
else:
    exit()

verifier = RobustnessVerifier if state_type == 'mixed' else PureRobustnessVerifier

# ac, time = PrettyTable(), PrettyTable()
# ac.add_column('epsilon', ['Rough Verif', 'Accurate Verif'])
# time.add_column('epsilon', ['Rough Verif', 'Accurate Verif'])
res_table = PrettyTable(['epsilon', 'Circuit',
                         'Rough Verif RA(%)', 'Rough Verif VT(s)',
                         'Accurate Verif RA(%)', 'Accurate Verif VT(s)'])
for j in range(batch_num):
    c_eps = epsilon * (j + 1)
    for kraus_temp in [kraus, random_kraus, final_kraus]:
        if GET_NEW_DATASET:
            if 'mnist' in data_file:
                acc, time, _, new_data, new_labels = verifier(kraus_temp, O, data, label, c_eps, type,
                                                              GET_NEW_DATASET, origin_dataset_size,
                                                              ADVERSARY_EXAMPLE, digits, 'mnist')
            else:
                acc, time, _, new_data, new_labels = verifier(kraus_temp, O, data, label, c_eps, type,
                                                              GET_NEW_DATASET, origin_dataset_size)
            np.savez('./model_and_data/newdata_for_AT/{}_by_{}.npz'.format(model_name, c_eps),
                     O=O, data=new_data, label=new_labels, kraus=kraus_temp)
            print(result_file_name + " was saved successfully! ")
        else:
            if 'mnist' in data_file:
                acc, time = verifier(kraus_temp, O, data, label, c_eps, type,
                                     GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE,
                                     digits, 'mnist')
            else:
                acc, time = verifier(kraus_temp, O, data, label, c_eps, type,
                                     GET_NEW_DATASET, origin_dataset_size)
        if kraus_temp.shape == kraus.shape and (kraus_temp == kraus).all():
            res_table.add_row(['{:e}'.format(c_eps), 'noiseless',
                               '{:.2f}'.format(acc[0] * 100), '{:.4f}'.format(time[0]),
                               '{:.2f}'.format(acc[1] * 100), '{:.4f}'.format(time[1])])
        elif kraus_temp.shape == random_kraus.shape and (kraus_temp == random_kraus).all():
            res_table.add_row(['{:e}'.format(c_eps), 'random noise',
                               '{:.2f}'.format(acc[0] * 100), '{:.4f}'.format(time[0]),
                               '{:.2f}'.format(acc[1] * 100), '{:.4f}'.format(time[1])])
        else:  # 'random & specified noise'
            res_table.add_row(['{:e}'.format(c_eps), 'random & {}_{}'.format(noise_type.replace('_', '-'), noise_p),
                               '{:.2f}'.format(acc[0] * 100), '{:.4f}'.format(time[0]),
                               '{:.2f}'.format(acc[1] * 100), '{:.4f}'.format(time[1])])

file_path = './results/result_tables/' + result_file_name
with open(file_path, 'w', newline='') as f_output:
    # f_output.write(ac.get_csv_string())
    # f_output.write('\n')
    # f_output.write(time.get_csv_string())
    # f_output.close()
    f_output.write(res_table.get_csv_string())
    print('Got the result table: ')
    print(res_table)
    print(result_file_name + " was saved successfully! ")
