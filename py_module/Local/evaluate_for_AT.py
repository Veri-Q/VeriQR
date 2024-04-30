from VeriL import RobustnessVerifier, PureRobustnessVerifier
from common_interface import *
from numpy import load
from sys import argv
import csv
import gc
import copy


def adversarial_training_evaluate_mnist():
    n = 1
    state_flag = 'pure'
    ADVERSARY_EXAMPLE = True

    verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

    for d0 in range(0, 10):
        for d1 in range(d0 + 1, 10):
            digits = str(d0) + str(d1)
            if digits in ['13']:
                continue

            data_file = './model_and_data/mnist{}_data.npz'.format(digits)
            # data_file = './model_and_data/TFIchain8_data.npz'
            model_name = data_file[data_file.rfind('/') + 1: data_file.rfind('_')]
            DATA = load(data_file)
            O = DATA['O']
            data = DATA['data']
            label = DATA['label']

            noise_type = random.choice(["bit_flip", "depolarizing", "phase_flip", "mixed"])
            iter_num = 2

            noise_p = random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075])
            eps = random.choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01])
            type = 'qasm'
            # eps = choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075])

            print('noise_type =', noise_type)
            print('noise_p =', noise_p)
            print('eps =', eps)

            noise_list = []
            kraus_file = None
            if noise_type == 'mixed':
                noise_list = ["bit_flip", "depolarizing", "phase_flip"]

            qasm_file = './model_and_data/' + model_name + '.qasm'
            origin_circuit, origin_kraus = qasm2mq_with_kraus(qasm_file)
            random_circuit, random_kraus = generating_circuit_with_random_noise(origin_circuit, model_name)
            final_kraus, noise_name = generating_circuit_with_specified_noise(random_circuit, random_kraus, noise_type,
                                                                              noise_list,
                                                                              kraus_file, noise_p, model_name)

            noise_ = noise_type.replace('_', '-')
            with open("./results/adversarial_training.csv", "a+") as csvfile:
                w = csv.writer(csvfile)
                c_eps = eps
                for i in range(iter_num):
                    print('kraus.shape =', final_kraus.shape)
                    print('data.shape =', data.shape)
                    print('label.shape =', label.shape)
                    print('O.shape =', O.shape)
                    # origin_ac_temp, origin_time_temp, new_labels = verifier(origin_kraus, O, data, label, c_eps, type,
                    #                                                         ADVERSARY_EXAMPLE, digits, 'mnist')
                    # random_ac_temp, random_time_temp, new_labels = verifier(random_kraus, O, data, label, c_eps, type,
                    #                                                         ADVERSARY_EXAMPLE, digits, 'mnist')
                    final_ac_temp, final_time_temp, new_data, new_labels = verifier(final_kraus, O, data, label, c_eps,
                                                                                    type,
                                                                                    ADVERSARY_EXAMPLE, digits, 'mnist')

                    data = new_data
                    label = new_labels

                    # origin_ac_1 = origin_ac_temp[0] * 100
                    # origin_ac_2 = origin_ac_temp[1] * 100
                    # origin_time_1 = origin_time_temp[0]
                    # origin_time_2 = origin_time_temp[1]
                    #
                    # random_ac_1 = random_ac_temp[0] * 100
                    # random_ac_2 = random_ac_temp[1] * 100
                    # random_time_1 = random_time_temp[0]
                    # random_time_2 = random_time_temp[1]

                    final_ac_1 = final_ac_temp[0] * 100
                    final_ac_2 = final_ac_temp[1] * 100
                    final_time_1 = final_time_temp[0]
                    final_time_2 = final_time_temp[1]
                    w.writerow([model_name, '{}_{}_{}'.format(noise_, noise_p, c_eps),
                                'V_{}'.format(i),
                                '%.2f' % final_ac_1, '%.4f' % final_time_1,
                                '%.2f' % final_ac_2, '%.4f' % final_time_2])


def adversarial_training_evaluate():
    iter_num = 2
    GET_NEW_DATASET = True

    if '.npz' in str(argv[1]):
        # for example:
        # python evaluate.py ./model_and_data/qubit_cav.npz 0.001 1 mixed
        data_file = str(argv[1])
        eps = float(argv[2])
        n = int(argv[3])
        state_flag = str(argv[4])
        c_eps = eps

        DATA = load(data_file)
        kraus = DATA['kraus']
        O = DATA['O']
        data = DATA['data']
        label = DATA['label']
        type = 'npz'
        origin_dataset_size = label.shape[0]
        model_name = data_file[data_file.rfind('/') + 1: data_file.rfind('_')]
        file_name = '{}_{}_{}_{}.csv'.format(model_name, eps, n, state_flag)  # 默认文件名

        verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

        with open("./results/adversarial_training.csv", "a+") as csvfile:
            w = csv.writer(csvfile)
            for i in range(iter_num):
                final_ac_temp, final_time_temp, new_data, new_labels, non_robust_num_1, non_robust_num_2 = \
                    verifier(kraus, O, data, label, c_eps, type, GET_NEW_DATASET, origin_dataset_size)
                data = copy.deepcopy(new_data)
                label = copy.deepcopy(new_labels)
                final_ac_1 = final_ac_temp[0] * 100
                final_ac_2 = final_ac_temp[1] * 100
                final_time_1 = final_time_temp[0]
                final_time_2 = final_time_temp[1]
                # noise_ = noise_type.replace('_', '-')
                w.writerow([model_name, c_eps, 'c_0', 'V_{}'.format(i),
                            non_robust_num_1, '%.2f' % final_ac_1, '%.4f' % final_time_1,
                            non_robust_num_2, '%.2f' % final_ac_2, '%.4f' % final_time_2])

            data = DATA['data']
            label = DATA['label']
            for i in range(iter_num):
                final_ac_temp, final_time_temp, new_data, new_labels, non_robust_num_1, non_robust_num_2 = \
                    verifier(kraus, O, data, label, c_eps, type, GET_NEW_DATASET, origin_dataset_size)
                data = copy.deepcopy(new_data)
                label = copy.deepcopy(new_labels)
                final_ac_1 = final_ac_temp[0] * 100
                final_ac_2 = final_ac_temp[1] * 100
                final_time_1 = final_time_temp[0]
                final_time_2 = final_time_temp[1]
                w.writerow([model_name, c_eps, 'c_1', 'V_{}'.format(i),
                            non_robust_num_1, '%.2f' % final_ac_1, '%.4f' % final_time_1,
                            non_robust_num_2, '%.2f' % final_ac_2, '%.4f' % final_time_2])

            data = DATA['data']
            label = DATA['label']
            noise_ = 'depolarizing'
            noise_probs = [0.001, 0.005]
            for noise_p in noise_probs:
                for i in range(iter_num):
                    final_ac_temp, final_time_temp, new_data, new_labels, non_robust_num_1, non_robust_num_2 = \
                        verifier(kraus, O, data, label, c_eps, type, GET_NEW_DATASET, origin_dataset_size)
                    data = copy.deepcopy(new_data)
                    label = copy.deepcopy(new_labels)
                    final_ac_1 = final_ac_temp[0] * 100
                    final_ac_2 = final_ac_temp[1] * 100
                    final_time_1 = final_time_temp[0]
                    final_time_2 = final_time_temp[1]
                    w.writerow([model_name, c_eps, 'c_2 ({}_{})'.format(noise_, noise_p), 'V_{}'.format(i),
                                non_robust_num_1, '%.2f' % final_ac_1, '%.4f' % final_time_1,
                                non_robust_num_2, '%.2f' % final_ac_2, '%.4f' % final_time_2])

    else:
        # '.qasm' in str(argv[1])
        qasm_file = str(argv[1])
        data_file = str(argv[2])
        state_flag = str(argv[3])
        noise_type = argv[5]

        model_name = qasm_file[qasm_file.rfind('/') + 1:-5]
        verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

        digits = '36'
        ADVERSARY_EXAMPLE = False
        if 'mnist' in data_file:  # digits != '36'
            ADVERSARY_EXAMPLE = (str(argv[4]) == 'true')
            if '_data' in data_file:  # digits != '36'
                digits = data_file[data_file.rfind('_data') - 2: data_file.rfind('_data')]

        noise_list = []
        kraus_file = None
        if noise_type == 'mixed':
            noise_list = ["bit_flip", "depolarizing", "phase_flip"]
        # arg_num = len(argv)
        # if noise_type == 'mixed':
        #     noise_list = [i for i in argv[8: arg_num - 1]]
        #     print("noise_list: ", noise_list)
        # elif noise_type == 'custom':
        #     kraus_file = argv[8]
        origin_circuit, origin_kraus = qasm2mq_with_kraus(qasm_file)
        # if model_name not in ['mnist13', 'tfi4', 'tfi8', 'fashion10']:
        # if model_name not in ['fashion10']:
        random_circuit, random_kraus = generating_circuit_with_random_noise(origin_circuit, model_name)

        # probs_iris = [0.01, 0.01, 0.05, 0.05]
        # epss_iris = [0.005, 0.05, 0.003, 0.01]
        # probs_fashion8 = [0.01, 0.01, 0.05, 0.05]
        # epss_fashion8 = [0.001, 0.01, 0.001, 0.005]
        # probs_fashion10 = [0.001]
        # epss_fashion10 = [0.0001]
        # probs_mnist13 = [0.001, 0.001, 0.01, 0.01]
        # epss_mnist13 = [0.001, 0.003, 0.0001, 0.001]
        # probs_tfi4 = [0.01, 0.01, 0.05, 0.05]
        # epss_tfi4 = [0.001, 0.005, 0.005, 0.01]
        # probs_tfi8 = [0.01, 0.01, 0.05, 0.05]
        # epss_tfi8 = [0.001, 0.01, 0.005, 0.01]
        # probs_tfi12 = [0.1, 0.15, 0.03, 0.175]
        # epss_tfi12 = [0.003, 0.005, 0.01, 0.0001]

        probs_iris = [0.01, 0.05]
        epss_iris = 0.005
        probs_fashion8 = [0.01, 0.05]
        epss_fashion8 = 0.001
        probs_fashion10 = [0.001, 0.005]
        epss_fashion10 = 0.0001
        probs_mnist13 = [0.001, 0.01]
        epss_mnist13 = 0.003
        probs_tfi4 = [0.01, 0.05]
        epss_tfi4 = 0.005
        probs_tfi8 = [0.01, 0.05]
        epss_tfi8 = 0.001

        prob_map = {
            'iris': probs_iris,
            'fashion8': probs_fashion8,
            'fashion10': probs_fashion10,
            'mnist13': probs_mnist13,
            'tfi4': probs_tfi4,
            'tfi8': probs_tfi8,
            # 'tfi12': probs_tfi12,
        }
        eps_map = {
            'iris': epss_iris,
            'fashion8': epss_fashion8,
            'fashion10': epss_fashion10,
            'mnist13': epss_mnist13,
            'tfi4': epss_tfi4,
            'tfi8': epss_tfi8,
            # 'tfi12': epss_tfi12,
        }
        probs = prob_map[model_name]
        c_eps = eps_map[model_name]
        prob_1 = probs[0]
        prob_2 = probs[1]
        if model_name == "fashion10":
            np.savez('./fashion10_random_kraus.npz', random_kraus=random_kraus)
            # np.savez('./fashion10_final_kraus.npz', final_kraus=final_kraus)
            # random_kraus = load('./fashion10_random_kraus.npz')['random_kraus']
            # random_circuit = origin_circuit
            # final_kraus = load('./fashion10_final_kraus.npz')['final_kraus']
        # elif model_name not in ['mnist13', 'tfi4', 'tfi8']:
        final_kraus_1, noise_name = generating_circuit_with_specified_noise(random_circuit, random_kraus,
                                                                            noise_type, noise_list,
                                                                            kraus_file, prob_1, model_name)
        final_kraus_2, noise_name = generating_circuit_with_specified_noise(random_circuit, random_kraus,
                                                                            noise_type, noise_list,
                                                                            kraus_file, prob_2, model_name)
        DATA = load(data_file)
        O = DATA['O']
        data = DATA['data']
        label = DATA['label']
        type = 'qasm'

        origin_dataset_size = label.shape[0]
        noise_ = noise_type.replace('_', '-')
        # data_c0, data_c1, data_c2_1, data_c2_2 = data, data, data, data
        # label_c0, label_c1, label_c2_1, label_c2_2 = label, label, label, label
        data_c0, label_c0 = copy.deepcopy(data), copy.deepcopy(label)
        data_c1, label_c1 = copy.deepcopy(data), copy.deepcopy(label)
        data_c2_1, label_c2_1 = copy.deepcopy(data), copy.deepcopy(label)
        data_c2_2, label_c2_2 = copy.deepcopy(data), copy.deepcopy(label)
        with (open("./results/adversarial_training.csv", "a+") as csvfile):
            w = csv.writer(csvfile)
            for i in range(iter_num):
                if 'mnist' in model_name:
                    origin_ac_temp, origin_time_temp, new_data_c0, new_labels_c0, non_robust_num_1_c0, non_robust_num_2_c0 = \
                        verifier(origin_kraus, O, data_c0, label_c0, c_eps, type,
                                 GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
                    random_ac_temp, random_time_temp, new_data_c1, new_labels_c1, non_robust_num_1_c1, non_robust_num_2_c1 = \
                        verifier(random_kraus, O, data_c1, label_c1, c_eps, type,
                                 GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
                    final_ac_temp_1, final_time_temp_1, new_data_c2_1, new_labels_c2_1, non_robust_num_1_c21, non_robust_num_2_c21 = \
                        verifier(final_kraus_1, O, data_c2_1, label_c2_1, c_eps, type,
                                 GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
                    final_ac_temp_2, final_time_temp_2, new_data_c2_2, new_labels_c2_2, non_robust_num_1_c22, non_robust_num_2_c22 = \
                        verifier(final_kraus_2, O, data_c2_2, label_c2_2, c_eps, type,
                                 GET_NEW_DATASET, origin_dataset_size, ADVERSARY_EXAMPLE, digits, 'mnist')
                # elif model_name in ['tfi4', 'tfi8']:
                #     final_ac_temp, final_time_temp, new_data_c0, new_labels_c0 = verifier(origin_kraus, O, data_c0,
                #                                                                           label_c0,
                #                                                                           c_eps, type)
                else:
                    origin_ac_temp, origin_time_temp, new_data_c0, new_labels_c0, non_robust_num_1_c0, non_robust_num_2_c0 = \
                        verifier(origin_kraus, O, data_c0, label_c0, c_eps, type, GET_NEW_DATASET, origin_dataset_size)
                    random_ac_temp, random_time_temp, new_data_c1, new_labels_c1, non_robust_num_1_c1, non_robust_num_2_c1 = \
                        verifier(random_kraus, O, data_c1, label_c1, c_eps, type, GET_NEW_DATASET, origin_dataset_size)
                    final_ac_temp_1, final_time_temp_1, new_data_c2_1, new_labels_c2_1, non_robust_num_1_c21, non_robust_num_2_c21 = \
                        verifier(final_kraus_1, O, data_c2_1, label_c2_1, c_eps, type, GET_NEW_DATASET,
                                 origin_dataset_size)
                    final_ac_temp_2, final_time_temp_2, new_data_c2_2, new_labels_c2_2, non_robust_num_1_c22, non_robust_num_2_c22 = \
                        verifier(final_kraus_2, O, data_c2_2, label_c2_2, c_eps, type, GET_NEW_DATASET,
                                 origin_dataset_size)

                data_c0, label_c0 = copy.deepcopy(new_data_c0), copy.deepcopy(new_labels_c0)
                data_c1, label_c1 = copy.deepcopy(new_data_c1), copy.deepcopy(new_labels_c1)
                data_c2_1, label_c2_1 = copy.deepcopy(new_data_c2_1), copy.deepcopy(new_labels_c2_1)
                data_c2_2, label_c2_2 = copy.deepcopy(new_data_c2_2), copy.deepcopy(new_labels_c2_2)
                np.savez('model_and_data/by_adversarial_training/iris_newdata_c0.npz', O=O, data=new_data_c0,
                         label=new_labels_c0)
                np.savez('model_and_data/by_adversarial_training/iris_newdata_c1.npz', O=O, data=new_data_c1,
                         label=new_labels_c1)
                np.savez('model_and_data/by_adversarial_training/iris_newdata_c21.npz', O=O, data=new_data_c2_1,
                         label=new_labels_c2_1)
                np.savez('model_and_data/by_adversarial_training/iris_newdata_c22.npz', O=O, data=new_data_c2_2,
                         label=new_labels_c2_2)

                origin_ac_1 = origin_ac_temp[0] * 100
                origin_ac_2 = origin_ac_temp[1] * 100
                origin_time_1 = origin_time_temp[0]
                origin_time_2 = origin_time_temp[1]

                random_ac_1 = random_ac_temp[0] * 100
                random_ac_2 = random_ac_temp[1] * 100
                random_time_1 = random_time_temp[0]
                random_time_2 = random_time_temp[1]

                final_ac_1_c21 = final_ac_temp_1[0] * 100
                final_ac_2_c21 = final_ac_temp_1[1] * 100
                final_time_1_c21 = final_time_temp_1[0]
                final_time_2_c21 = final_time_temp_1[1]

                final_ac_1_c22 = final_ac_temp_2[0] * 100
                final_ac_2_c22 = final_ac_temp_2[1] * 100
                final_time_1_c22 = final_time_temp_2[0]
                final_time_2_c22 = final_time_temp_2[1]

                # w.writerow([model_name, '{}_{}_{}'.format(noise_, noise_p, c_eps),
                #             'V_{}'.format(i),
                #             '%.2f' % final_ac_1, '%.4f' % final_time_1,
                #             '%.2f' % final_ac_2, '%.4f' % final_time_2])
                # w.writerows([
                #     [model_name, '{}_{}_{}'.format(noise_, noise_p, c_eps), 'c_0', 'V_{}'.format(i),
                #      '%.2f' % origin_ac_1, '%.4f' % origin_time_1, '%.2f' % origin_ac_2, '%.4f' % origin_time_2],
                #     [model_name, '{}_{}_{}'.format(noise_, noise_p, c_eps), 'c_1', 'V_{}'.format(i),
                #      '%.2f' % random_ac_1, '%.4f' % random_time_1, '%.2f' % random_ac_2, '%.4f' % random_time_2],
                #     [model_name, '{}_{}_{}'.format(noise_, noise_p, c_eps), 'c_2', 'V_{}'.format(i),
                #      '%.2f' % final_ac_1, '%.4f' % final_time_1, '%.2f' % final_ac_2, '%.4f' % final_time_2]
                # ])
                w.writerows([
                    [model_name, c_eps, 'c_0', 'V_{}'.format(i),
                     non_robust_num_1_c0, '%.2f' % origin_ac_1, '%.4f' % origin_time_1,
                     non_robust_num_2_c0, '%.2f' % origin_ac_2, '%.4f' % origin_time_2],
                    [model_name, c_eps, 'c_1', 'V_{}'.format(i),
                     non_robust_num_1_c1, '%.2f' % random_ac_1, '%.4f' % random_time_1,
                     non_robust_num_2_c1, '%.2f' % random_ac_2, '%.4f' % random_time_2],
                    [model_name, c_eps, 'c_2 ({}_{})'.format(noise_, prob_1), 'V_{}'.format(i),
                     non_robust_num_1_c21, '%.2f' % final_ac_1_c21, '%.4f' % final_time_1_c21,
                     non_robust_num_2_c21, '%.2f' % final_ac_2_c21, '%.4f' % final_time_2_c21],
                    [model_name, c_eps, 'c_2 ({}_{})'.format(noise_, prob_2), 'V_{}'.format(i),
                     non_robust_num_1_c22, '%.2f' % final_ac_1_c22, '%.4f' % final_time_1_c22,
                     non_robust_num_2_c22, '%.2f' % final_ac_2_c22, '%.4f' % final_time_2_c22]
                ])

# adversarial_training_evaluate()
# gc.collect()
