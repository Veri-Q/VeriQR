from VeriL import RobustnessVerifier, PureRobustnessVerifier
from common_interface import *
from numpy import load
from sys import argv
import csv
import gc
import copy
import os


# def adversarial_training_evaluate_mnist():
#     n = 1
#     state_flag = 'pure'
#     ADVERSARY_EXAMPLE = True
#     iter_num = 2
#     GET_NEW_DATASET = False
#
#     verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier
#
#     for d0 in range(0, 10):
#         for d1 in range(d0 + 1, 10):
#             digits = str(d0) + str(d1)
#             if digits in ['13']:
#                 continue
#
#             data_file = './model_and_data/mnist{}_data.npz'.format(digits)
#             # data_file = './model_and_data/TFIchain8_data.npz'
#             model_name = data_file[data_file.rfind('/') + 1: data_file.rfind('_')]
#             DATA = load(data_file)
#             O = DATA['O']
#             data = DATA['data']
#             label = DATA['label']
#
#             noise_type = random.choice(["bit_flip", "depolarizing", "phase_flip", "mixed"])
#             iter_num = 2
#
#             noise_p = random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075])
#             eps = random.choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01])
#             type = 'qasm'
#             # eps = choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075])
#
#             print('noise_type =', noise_type)
#             print('noise_p =', noise_p)
#             print('eps =', eps)
#
#             noise_list = []
#             kraus_file = None
#             if noise_type == 'mixed':
#                 noise_list = ["bit_flip", "depolarizing", "phase_flip"]
#
#             qasm_file = './model_and_data/' + model_name + '.qasm'
#             origin_circuit, origin_kraus = qasm2mq_with_kraus(qasm_file)
#             random_circuit, random_kraus = generating_circuit_with_random_noise(origin_circuit, model_name)
#             final_kraus, noise_name = generating_circuit_with_specified_noise(random_circuit, random_kraus, noise_type,
#                                                                               noise_list,
#                                                                               kraus_file, noise_p, model_name)
#
#             noise_ = noise_type.replace('_', '-')
#             with open("./results/adversarial_training.csv", "a+") as csvfile:
#                 w = csv.writer(csvfile)
#                 c_eps = eps
#                 for i in range(iter_num):
#                     print('kraus.shape =', final_kraus.shape)
#                     print('data.shape =', data.shape)
#                     print('label.shape =', label.shape)
#                     print('O.shape =', O.shape)
#                     # origin_ac_temp, origin_time_temp, new_labels = verifier(origin_kraus, O, data, label, c_eps, type,
#                     #                                                         ADVERSARY_EXAMPLE, digits, 'mnist')
#                     # random_ac_temp, random_time_temp, new_labels = verifier(random_kraus, O, data, label, c_eps, type,
#                     #                                                         ADVERSARY_EXAMPLE, digits, 'mnist')
#                     final_ac_temp, final_time_temp, new_data, new_labels = verifier(final_kraus, O, data, label, c_eps,
#                                                                                     type,
#                                                                                     ADVERSARY_EXAMPLE, digits, 'mnist')
#
#                     data = new_data
#                     label = new_labels
#
#                     # origin_ac_1 = origin_ac_temp[0] * 100
#                     # origin_ac_2 = origin_ac_temp[1] * 100
#                     # origin_time_1 = origin_time_temp[0]
#                     # origin_time_2 = origin_time_temp[1]
#                     #
#                     # random_ac_1 = random_ac_temp[0] * 100
#                     # random_ac_2 = random_ac_temp[1] * 100
#                     # random_time_1 = random_time_temp[0]
#                     # random_time_2 = random_time_temp[1]
#
#                     final_ac_1 = final_ac_temp[0] * 100
#                     final_ac_2 = final_ac_temp[1] * 100
#                     final_time_1 = final_time_temp[0]
#                     final_time_2 = final_time_temp[1]
#                     w.writerow([model_name, '{}_{}_{}'.format(noise_, noise_p, c_eps),
#                                 'V_{}'.format(i),
#                                 '%.2f' % final_ac_1, '%.4f' % final_time_1,
#                                 '%.2f' % final_ac_2, '%.4f' % final_time_2])


def generate_newdata_for_adversarial_training():
    GET_NEW_DATASET = True

    if '.npz' in str(argv[1]):
        # for example:
        # python evaluate.py ./model_and_data/qubit_cav.npz mixed 0.001 0.005
        data_file = str(argv[1])
        state_flag = str(argv[2])
        epss = [float(argv[i]) for i in range(3, len(argv))]

        DATA = load(data_file)
        kraus = DATA['kraus']
        O = DATA['O']
        data = DATA['data']
        label = DATA['label']
        type_ = 'npz'
        origin_dataset_size = label.shape[0]
        model_name = data_file[data_file.rfind('/') + 1: data_file.rfind('_')]

        digits = '36'
        ADVERSARY_EXAMPLE = False
        TEST_MNIST = ('mnist' in data_file)

        verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

        np.savez('./model_and_data/origin_model/{}_c0.npz'.format(model_name), kraus=kraus)  # iris_c0

        kraus_file = None
        probs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
        noise_types = ["bit_flip", "depolarizing", "phase_flip", "mixed"]
        for noise_type in noise_types:
            gc.collect()
            noise_list = ["bit_flip", "depolarizing", "phase_flip"] if noise_type == 'mixed' else []
            p = random.choice(probs)
            noise_name = noise_name_map[noise_type]
            final_kraus, _ = generating_circuit_with_specified_noise(Circuit(), kraus,
                                                                  noise_type, noise_list,
                                                                  kraus_file, p, model_name)
            np.savez('./model_and_data/origin_model/{}_c2_{}_{}.npz'.format(model_name, noise_name, p),
                     kraus=final_kraus)  # iris_c0
            for c_eps in epss:
                if TEST_MNIST:
                    final_acc, final_time, non_robust_num_c2, new_data_c2, new_labels_c2 = \
                        verifier(final_kraus, O, data, label, c_eps, type_,
                                 GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
                else:
                    final_acc, final_time, non_robust_num_c2, new_data_c2, new_labels_c2 = \
                        verifier(final_kraus, O, data, label, c_eps, type_, GET_NEW_DATASET,
                                 origin_dataset_size)

                np.savez(
                    './model_and_data/newdata_for_AT/{}_c2_{}_{}_by_{}.npz'.format(model_name, noise_name, p, c_eps),
                    O=O, data=new_data_c2, label=new_labels_c2)
                final_ac_1 = final_acc[0] * 100
                final_ac_2 = final_acc[1] * 100
                noise_ = noise_type.replace('_', '-')
                with open("./results/adversarial_training.csv", "a+") as csvfile:
                    w = csv.writer(csvfile)
                    w.writerow([model_name, 'c_2', '{}_{}'.format(noise_, p), c_eps, 'before',
                                non_robust_num_c2[0], '%.2f' % final_ac_1, '%.4f' % final_time[0],
                                non_robust_num_c2[1], '%.2f' % final_ac_2, '%.4f' % final_time[1]])
    else:
        # python evaluate_for_AT.py ./model_and_data/fashion8.qasm ./model_and_data/fashion8_data.npz pure 0.001 0.005
        qasm_file = str(argv[1])
        data_file = str(argv[2])
        state_flag = str(argv[3])
        epss = [float(argv[i]) for i in range(4, len(argv))]
        # epss = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]

        model_name = qasm_file[qasm_file.rfind('/') + 1:-5]
        verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

        digits = '36'
        ADVERSARY_EXAMPLE = False
        TEST_MNIST = ('mnist' in data_file)
        if TEST_MNIST:  # digits != '36'
            if '_data' in data_file:  # digits != '36'
                digits = data_file[data_file.rfind('_data') - 2: data_file.rfind('_data')]

        DATA = load(data_file)
        O = DATA['O']
        data = DATA['data']
        label = DATA['label']
        type_ = 'qasm'
        origin_dataset_size = label.shape[0]

        file_path = './model_and_data/origin_model/'

        file_name = file_path + '{}_c0'.format(model_name)
        origin_circuit, origin_kraus = qasm2mq_with_kraus(qasm_file, True, file_name + '.svg')
        np.savez(file_name + '.npz', kraus=origin_kraus)  # iris_c0
        for c_eps in epss:
            gc.collect()
            if TEST_MNIST:
                # origin_acc, origin_time, non_robust_num_c0, new_data_c0, new_labels_c0 = \
                #     verifier(origin_kraus, O, data_c0, label_c0, c_eps, type,
                #              GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
                origin_acc, origin_time, non_robust_num_c0, new_data_c0, new_labels_c0 = \
                    verifier(origin_kraus, O, data, label, c_eps, type_,
                             GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
            else:
                # origin_acc, origin_time, non_robust_num_c0, new_data_c0, new_labels_c0 = \
                #     verifier(origin_kraus, O, data_c0, label_c0, c_eps, type,
                #              GET_NEW_DATASET, origin_dataset_size)
                origin_acc, origin_time, non_robust_num_c0, new_data_c0, new_labels_c0 = \
                    verifier(origin_kraus, O, data, label, c_eps, type_, GET_NEW_DATASET, origin_dataset_size)

            np.savez('./model_and_data/newdata_for_AT/{}_c0_by_{}.npz'.format(model_name, c_eps),
                     O=O, data=new_data_c0, label=new_labels_c0)
            # data_c0, label_c0 = copy.deepcopy(new_data_c0), copy.deepcopy(new_labels_c0)
            # data_c1, label_c1 = copy.deepcopy(new_data_c1), copy.deepcopy(new_labels_c1)
            origin_ac_1 = origin_acc[0] * 100
            origin_ac_2 = origin_acc[1] * 100
            with open("./results/adversarial_training.csv", "a+") as csvfile:
                w = csv.writer(csvfile)
                w.writerow([model_name, 'c_0', 'noiseless', c_eps, 'before',
                            non_robust_num_c0[0], '%.2f' % origin_ac_1, '%.4f' % origin_time[0],
                            non_robust_num_c0[1], '%.2f' % origin_ac_2, '%.4f' % origin_time[1]])

        file_name = file_name.replace('c0', 'c1')
        random_circuit, random_kraus = generating_circuit_with_random_noise(origin_circuit, model_name,
                                                                            True, file_name + '.svg')
        np.savez(file_name + '.npz', kraus=random_kraus)  # iris_c1
        for c_eps in epss:
            gc.collect()
            if TEST_MNIST:
                # random_acc, random_time, non_robust_num_c1, new_data_c1, new_labels_c1 = \
                #     verifier(random_kraus, O, data_c1, label_c1, c_eps, type,
                #              GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
                random_acc, random_time, non_robust_num_c1, new_data_c1, new_labels_c1 = \
                    verifier(random_kraus, O, data, label, c_eps, type_,
                             GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
            else:
                # random_acc, random_time, non_robust_num_c1, new_data_c1, new_labels_c1 = \
                #     verifier(random_kraus, O, data_c1, label_c1, c_eps, type,
                #              GET_NEW_DATASET, origin_dataset_size)
                random_acc, random_time, non_robust_num_c1, new_data_c1, new_labels_c1 = \
                    verifier(random_kraus, O, data, label, c_eps, type_,
                             GET_NEW_DATASET, origin_dataset_size)

            np.savez('./model_and_data/newdata_for_AT/{}_c1_by_{}.npz'.format(model_name, c_eps),
                     O=O, data=new_data_c1, label=new_labels_c1)

            random_ac_1 = random_acc[0] * 100
            random_ac_2 = random_acc[1] * 100
            with open("./results/adversarial_training.csv", "a+") as csvfile:
                w = csv.writer(csvfile)
                w.writerow([model_name, 'c_1', 'random', c_eps, 'before',
                            non_robust_num_c1[0], '%.2f' % random_ac_1, '%.4f' % random_time[0],
                            non_robust_num_c1[1], '%.2f' % random_ac_2, '%.4f' % random_time[1]])

        kraus_file = None
        probs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
        noise_types = ["bit_flip", "depolarizing", "phase_flip", "mixed"]
        for noise_type in noise_types:
            gc.collect()
            noise_list = ["bit_flip", "depolarizing", "phase_flip"] if noise_type == 'mixed' else []
            p = random.choice(probs)
            noise_name = noise_name_map[noise_type]
            file_name = file_path + '{}_c2_{}_{}'.format(model_name, noise_name, p)
            final_kraus, _ = generating_circuit_with_specified_noise(random_circuit, random_kraus,
                                                                  noise_type, noise_list,
                                                                  kraus_file, p, model_name,
                                                                  True, file_name + '.svg')
            np.savez(file_name + '.npz', kraus=final_kraus)  # iris_c1
            # ansatz_str = OpenQASM().to_string(random_circuit)
            # f = open(file_name + '.qasm', 'w')
            # f.write(ansatz_str)
            # f.close()
            for c_eps in epss:
                if TEST_MNIST:
                    final_acc, final_time, non_robust_num_c2, new_data_c2, new_labels_c2 = \
                        verifier(final_kraus, O, data, label, c_eps, type_,
                                 GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
                else:
                    final_acc, final_time, non_robust_num_c2, new_data_c2, new_labels_c2 = \
                        verifier(final_kraus, O, data, label, c_eps, type_, GET_NEW_DATASET,
                                 origin_dataset_size)

                np.savez(
                    './model_and_data/newdata_for_AT/{}_c2_{}_{}_by_{}.npz'.format(model_name, noise_name, p, c_eps),
                    O=O, data=new_data_c2, label=new_labels_c2)
                final_ac_1 = final_acc[0] * 100
                final_ac_2 = final_acc[1] * 100
                noise_ = noise_type.replace('_', '-')
                with open("./results/adversarial_training.csv", "a+") as csvfile:
                    w = csv.writer(csvfile)
                    w.writerow([model_name, 'c_2', '{}_{}'.format(noise_, p), c_eps, 'before',
                                non_robust_num_c2[0], '%.2f' % final_ac_1, '%.4f' % final_time[0],
                                non_robust_num_c2[1], '%.2f' % final_ac_2, '%.4f' % final_time[1]])


def generate_newdata_from_randomModel(model_name, qubits_num, state_flag, epss):
    GET_NEW_DATASET = True
    type_ = 'qasm'

    digits = '36'
    ADVERSARY_EXAMPLE = False
    TEST_MNIST = ('mnist' in model_name)
    if TEST_MNIST:  # digits != '36'
        if '_data' in model_name:  # digits != '36'
            digits = model_name[5:]

    ORIGIN_DATA = load('./model_and_data/{}_data.npz'.format(model_name))
    O = ORIGIN_DATA['O']
    data = ORIGIN_DATA['data']
    label = ORIGIN_DATA['label']
    origin_dataset_size = label.shape[0]

    file_path = './model_and_data/origin_model/'

    file_name = file_path + '{}_c1'.format(model_name)  # random circuit
    # random_circuit = qasm2mq(file_name + '.qasm', True, file_name + '.svg')
    random_circuit = Circuit(gates=I.on(qubits_num - 1))
    random_kraus = load(file_name + '.npz')['kraus']
    print(random_kraus.shape)

    verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

    kraus_file = None
    probs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
    noise_types = ["bit_flip", "depolarizing", "phase_flip", "mixed"]
    # noise_types = ["mixed"]
    for noise_type in noise_types:
        gc.collect()
        noise_list = ["bit_flip", "depolarizing", "phase_flip"] if noise_type == 'mixed' else []
        p = random.choice(probs)
        noise_name = noise_name_map[noise_type]
        file_name = file_path + '{}_c2_{}_{}'.format(model_name, noise_name, p)
        # final_kraus, _ = generating_circuit_with_specified_noise(random_circuit, random_kraus,
        #                                                       noise_type, noise_list,
        #                                                       kraus_file, p, model_name,
        #                                                       True, file_name + '.svg')
        final_kraus, _ = generating_circuit_with_specified_noise(random_circuit, random_kraus,
                                                              noise_type, noise_list,
                                                              kraus_file, p, model_name)
        np.savez(file_name + '.npz', kraus=final_kraus)  # iris_c1
        # ansatz_str = OpenQASM().to_string(random_circuit)
        # f = open(file_name + '.qasm', 'w')
        # f.write(ansatz_str)
        # f.close()
        for c_eps in epss:
            if TEST_MNIST:
                final_acc, final_time, non_robust_num_c2, new_data_c2, new_labels_c2 = \
                    verifier(final_kraus, O, data, label, c_eps, type_,
                             GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
            else:
                final_acc, final_time, non_robust_num_c2, new_data_c2, new_labels_c2 = \
                    verifier(final_kraus, O, data, label, c_eps, type_, GET_NEW_DATASET,
                             origin_dataset_size)

            np.savez('./model_and_data/newdata_for_AT/{}_c2_{}_{}_by_{}.npz'.format(model_name, noise_name, p, c_eps),
                     O=O, data=new_data_c2, label=new_labels_c2)
            final_ac_1 = final_acc[0] * 100
            final_ac_2 = final_acc[1] * 100
            noise_ = noise_type.replace('_', '-')
            with open("./results/adversarial_training.csv", "a+") as csvfile:
                w = csv.writer(csvfile)
                w.writerow([model_name, 'c_2', '{}_{}'.format(noise_, p), c_eps, 'before',
                            non_robust_num_c2[0], '%.2f' % final_ac_1, '%.4f' % final_time[0],
                            non_robust_num_c2[1], '%.2f' % final_ac_2, '%.4f' % final_time[1]])


def verify_new_model(model_name, state_flag):
    GET_NEW_DATASET = True
    type_ = 'qasm'

    digits = '36'
    ADVERSARY_EXAMPLE = False
    TEST_MNIST = ('mnist' in model_name)
    if TEST_MNIST:  # digits != '36'
        if '_data' in model_name:  # digits != '36'
            digits = model_name[5:]

    ORIGIN_DATA = load('./model_and_data/{}_data.npz'.format(model_name))
    origin_dataset_size = ORIGIN_DATA['label'].shape[0]

    verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

    model_path = './model_and_data/newmodel_by_AT/'
    data_path = './model_and_data/newdata_for_AT/'
    model_list = os.listdir(model_path)
    model_list = sorted(model_list, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
    for file_name in model_list:
        if os.path.isfile(os.path.join(model_path, file_name)):
            pass
        if (not file_name.startswith(model_name) or os.path.splitext(file_name)[-1] != '.npz'
                or file_name.startswith(model_name + '_c0') or file_name.startswith(model_name + '_c1')
                or file_name.startswith(model_name + '_c1_Bit')):
            continue
        # 'fashion8_c0_by_0.001.npz'
        # 'fashion8_c1_by_0.001.npz'
        # 'fashion8_c2_BitFlip_0.001_by_0.001.npz'
        # 'fashion8_c2_mixed_BitFlip_Depolarizing_PhaseFlip_0.001_by_0.001.npz'
        # file_name = os.path.basename(file_name)
        print(file_name)
        MODEL = load(os.path.join(model_path, file_name))
        kraus = MODEL['kraus']
        DATA = load(os.path.join(data_path, file_name))
        O = DATA['O']
        data = DATA['data']
        label = DATA['label']
        args = file_name[:-4].split('_')
        # model_name = args[0]
        circ_type = args[1]
        c_eps = float(args[len(args) - 1])
        if circ_type == 'c0':
            noise_set = 'noiseless'
        elif circ_type == 'c1':
            noise_set = 'random'
        elif circ_type == 'c2':
            noise_type = noise_name_map_reverse[args[2]] if args[2] != 'mixed' else args[2]
            p = float(args[len(args) - 3])
            noise_ = noise_type.replace('_', '-')
            noise_set = '{}_{}'.format(noise_, p)
            print('noise type: ', noise_type)
            print('noise p: ', p)
        print('circ_type: ', circ_type)
        print('c_eps: ', c_eps)
        print('noise setting: ', noise_set)

        print('kraus.shape: ', kraus.shape)
        print('data.shape: ', data.shape)
        print('label.shape: ', label.shape)
        print('O.shape: ', O.shape)
        if TEST_MNIST:
            final_acc, final_time, non_robust_num_c2, _, _ = \
                verifier(kraus, O, data, label, c_eps, type_,
                         GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
        else:
            final_acc, final_time, non_robust_num_c2, _, _ = \
                verifier(kraus, O, data, label, c_eps, type_, GET_NEW_DATASET,
                         origin_dataset_size)
        final_ac_1 = final_acc[0] * 100
        final_ac_2 = final_acc[1] * 100
        with open("./results/adversarial_training.csv", "a+") as csvfile:
            w = csv.writer(csvfile)
            w.writerow([model_name, circ_type.replace('c', 'c_'), noise_set, c_eps, 'after',
                        non_robust_num_c2[0], '%.2f' % final_ac_1, '%.4f' % final_time[0],
                        non_robust_num_c2[1], '%.2f' % final_ac_2, '%.4f' % final_time[1]])


generate_newdata_for_adversarial_training()
# generate_newdata_from_randomModel('iris', 4, 'pure', [0.01, 0.05])
# generate_newdata_from_randomModel('fashion8', 8, 'pure', [0.001, 0.003])
# generate_newdata_from_randomModel('mnist13', 8, 'pure', [0.003, 0.005])
# verify_new_model('iris', 'mixed')
# verify_new_model('fashion8', 'pure')
# verify_new_model('mnist13', 'pure')
