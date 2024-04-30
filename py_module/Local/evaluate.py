from VeriL import RobustnessVerifier, PureRobustnessVerifier
from common_interface import *
from numpy import load
from sys import argv
import csv
import gc


def evaluate():
    iter_num = 2
    GET_NEW_DATASET = False

    if '.npz' in str(argv[1]):
        # for example:
        # python3 batch_check.py binary_cav.npz 0.001 1 mixed
        data_file = str(argv[1])
        eps = float(argv[2])
        # n = int(argv[3])
        state_flag = str(argv[4])

        DATA = load(data_file)
        kraus = DATA['kraus']
        O = DATA['O']
        data = DATA['data']
        label = DATA['label']
        type = 'npz'
        # origin_dataset_size = label.shape[0]
        model_name = data_file[data_file.rfind('/') + 1: data_file.rfind('_')]
        # file_name = '{}_{}_{}_{}.csv'.format(model_name, eps, n, state_flag)  # 默认文件名

        verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

        with open("./results/adversarial_training.csv", "a+") as csvfile:
            w = csv.writer(csvfile)
            c_eps = eps
            for i in range(iter_num):
                origin_ac_temp, origin_time_temp = verifier(kraus, O, data, label, c_eps, type)
                # verifier(kraus, O, data, label, c_eps, type, GET_NEW_DATASET, origin_dataset_size)
                origin_ac_1 = origin_ac_temp[0] * 100
                origin_ac_2 = origin_ac_temp[1] * 100
                origin_time_1 = origin_time_temp[0]
                origin_time_2 = origin_time_temp[1]
                w.writerow([model_name, c_eps, 'c_0',
                            '%.2f' % origin_ac_1, '%.4f' % origin_time_1,
                            '%.2f' % origin_ac_2, '%.4f' % origin_time_2])
                gc.collect()

            for i in range(iter_num):
                random_ac_temp, random_time_temp = verifier(kraus, O, data, label, c_eps, type)
                random_ac_1 = random_ac_temp[0] * 100
                random_ac_2 = random_ac_temp[1] * 100
                random_time_1 = random_time_temp[0]
                random_time_2 = random_time_temp[1]
                w.writerow([model_name, c_eps, 'c_1',
                            '%.2f' % random_ac_1, '%.4f' % random_time_1,
                            '%.2f' % random_ac_2, '%.4f' % random_time_2])
                gc.collect()

            noise_ = 'depolarizing'
            noise_probs = [0.001, 0.005]
            for noise_p in noise_probs:
                for i in range(iter_num):
                    final_ac_temp, final_time_temp = verifier(kraus, O, data, label, c_eps, type)
                    final_ac_1 = final_ac_temp[0] * 100
                    final_ac_2 = final_ac_temp[1] * 100
                    final_time_1 = final_time_temp[0]
                    final_time_2 = final_time_temp[1]
                    w.writerow([model_name, c_eps, 'c_2 ({}_{})'.format(noise_, noise_p),
                                '%.2f' % final_ac_1, '%.4f' % final_time_1,
                                '%.2f' % final_ac_2, '%.4f' % final_time_2])
                    gc.collect()
    else:
        # '.qasm' in str(argv[1])
        qasm_file = str(argv[1])
        data_file = str(argv[2])
        state_flag = str(argv[3])
        # noise_type = argv[5]
        # n = 1

        DATA = load(data_file)
        O = DATA['O']
        data = DATA['data']
        label = DATA['label']
        type = 'qasm'

        model_name = qasm_file[qasm_file.rfind('/') + 1:-5]
        verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier

        digits = '36'
        ADVERSARY_EXAMPLE = False
        TEST_MNIST = ('mnist' in data_file)
        if TEST_MNIST:  # digits != '36'
            # ADVERSARY_EXAMPLE = (str(argv[4]) == 'true')
            if '_data' in data_file:  # digits != '36'
                digits = data_file[data_file.rfind('_data') - 2: data_file.rfind('_data')]

        origin_circuit, origin_kraus = qasm2mq_with_kraus(qasm_file)
        if model_name != "fashion10":
            random_circuit, random_kraus = generating_circuit_with_random_noise(origin_circuit, model_name)

        epss = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]
        probs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
        noise_types = ["bit_flip", "depolarizing", "phase_flip", "mixed"]
        kraus_file = None
        with open("./results/local_results.csv", "a+") as csvfile:
            w = csv.writer(csvfile)
            for c_eps in epss:
                if TEST_MNIST:
                    origin_ac_temp, origin_time_temp = verifier(origin_kraus, O, data, label, c_eps, type,
                                                                GET_NEW_DATASET, 0, ADVERSARY_EXAMPLE,
                                                                digits, 'mnist')
                else:
                    origin_ac_temp, origin_time_temp = verifier(origin_kraus, O, data, label, c_eps, type,
                                                                GET_NEW_DATASET, 0)
                origin_ac_1 = origin_ac_temp[0] * 100
                origin_ac_2 = origin_ac_temp[1] * 100
                origin_time_1 = origin_time_temp[0]
                origin_time_2 = origin_time_temp[1]
                w.writerow([model_name, 'c_0', c_eps,
                            '%.2f' % origin_ac_1, '%.4f' % origin_time_1,
                            '%.2f' % origin_ac_2, '%.4f' % origin_time_2])
            for c_eps in epss:
                if TEST_MNIST:
                    random_ac_temp, random_time_temp = verifier(random_kraus, O, data, label, c_eps, type,
                                                                GET_NEW_DATASET, 0, ADVERSARY_EXAMPLE,
                                                                digits, 'mnist')
                else:
                    random_ac_temp, random_time_temp = verifier(random_kraus, O, data, label, c_eps, type,
                                                                GET_NEW_DATASET, 0)
                random_ac_1 = random_ac_temp[0] * 100
                random_ac_2 = random_ac_temp[1] * 100
                random_time_1 = random_time_temp[0]
                random_time_2 = random_time_temp[1]
                w.writerow([model_name, 'c_1', c_eps,
                            '%.2f' % random_ac_1, '%.4f' % random_time_1,
                            '%.2f' % random_ac_2, '%.4f' % random_time_2])

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
            # prob_map = {
            #     'iris': probs_iris,
            #     'fashion8': probs_fashion8,
            #     'fashion10': probs_fashion10,
            #     'mnist13': probs_mnist13,
            #     'tfi4': probs_tfi4,
            #     'tfi8': probs_tfi8,
            #     'tfi12': probs_tfi12,
            # }
            # eps_map = {
            #     'iris': epss_iris,
            #     'fashion8': epss_fashion8,
            #     'fashion10': epss_fashion10,
            #     'mnist13': epss_mnist13,
            #     'tfi4': epss_tfi4,
            #     'tfi8': epss_tfi8,
            #     'tfi12': epss_tfi12,
            # }
            # probs = prob_map[model_name]
            # epss = eps_map[model_name]

            # for noise_p, c_eps in zip(probs, epss):
            for noise_type in noise_types:
                noise_list = ["bit_flip", "depolarizing", "phase_flip"] if noise_type == 'mixed' else []
                for noise_p in probs:
                    if model_name == "fashion10":
                        # np.savez('./fashion10_random_kraus.npz', random_kraus=random_kraus)
                        # np.savez('./fashion10_final_kraus.npz', final_kraus=final_kraus)
                        random_circuit = None  # no need
                        final_circuit = None  # no need
                        random_kraus = load('./fashion10_random_kraus.npz')['random_kraus']
                        final_kraus = load('./fashion10_final_kraus.npz')['final_kraus']
                    else:
                        final_kraus, noise_name = generating_circuit_with_specified_noise(random_circuit, random_kraus,
                                                                                          noise_type, noise_list,
                                                                                          kraus_file, noise_p,
                                                                                          model_name)
                        noise_ = noise_type.replace('_', '-')
                        for c_eps in epss:
                            if TEST_MNIST:
                                final_ac_temp, final_time_temp = verifier(final_kraus, O, data, label, c_eps, type,
                                                                          GET_NEW_DATASET, 0, ADVERSARY_EXAMPLE,
                                                                          digits, 'mnist')
                            else:
                                final_ac_temp, final_time_temp = verifier(final_kraus, O, data, label, c_eps, type,
                                                                          GET_NEW_DATASET, 0)
                            final_ac_1 = final_ac_temp[0] * 100
                            final_ac_2 = final_ac_temp[1] * 100
                            final_time_1 = final_time_temp[0]
                            final_time_2 = final_time_temp[1]
                            w.writerow([model_name, 'c_2 ({}_{})'.format(noise_, noise_p), c_eps,
                                        '%.2f' % final_ac_1, '%.4f' % final_time_1,
                                        '%.2f' % final_ac_2, '%.4f' % final_time_2])
                            gc.collect()


def evaluate_mnist(digits):
    if digits in ['13']:
        return

    state_flag = 'pure'
    verifier = RobustnessVerifier if state_flag == 'mixed' else PureRobustnessVerifier
    ADVERSARY_EXAMPLE = False
    GET_NEW_DATASET = False
    noise_types = ["bit_flip", "depolarizing", "phase_flip", "mixed"]
    epss = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]
    probs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
    noise_types = ["bit_flip", "depolarizing", "phase_flip", "mixed"]

    data_file = './model_and_data/mnist{}_data.npz'.format(digits)
    model_name = data_file[data_file.rfind('/') + 1: data_file.rfind('_')]
    DATA = load(data_file)
    O = DATA['O']
    data = DATA['data']
    label = DATA['label']
    type = 'qasm'

    c_eps = random.choice(epss)
    kraus_file = None

    qasm_file = './model_and_data/' + model_name + '.qasm'
    origin_circuit, origin_kraus = qasm2mq_with_kraus(qasm_file)
    random_circuit, random_kraus = generating_circuit_with_random_noise(origin_circuit, model_name)

    origin_ac_temp, origin_time_temp = verifier(origin_kraus, O, data, label, c_eps, type,
                                                GET_NEW_DATASET, 0, ADVERSARY_EXAMPLE,
                                                digits, 'mnist')
    random_ac_temp, random_time_temp = verifier(random_kraus, O, data, label, c_eps, type,
                                                GET_NEW_DATASET, 0, ADVERSARY_EXAMPLE,
                                                digits, 'mnist')
    origin_ac_1 = origin_ac_temp[0] * 100
    origin_ac_2 = origin_ac_temp[1] * 100
    origin_time_1 = origin_time_temp[0]
    origin_time_2 = origin_time_temp[1]
    random_ac_1 = random_ac_temp[0] * 100
    random_ac_2 = random_ac_temp[1] * 100
    random_time_1 = random_time_temp[0]
    random_time_2 = random_time_temp[1]
    with open("./results/local_results.csv", "a+") as csvfile:
        w = csv.writer(csvfile)
        w.writerows([
            [model_name, 'c_0', c_eps,
             '%.2f' % origin_ac_1, '%.4f' % origin_time_1,
             '%.2f' % origin_ac_2, '%.4f' % origin_time_2],
            [model_name, 'c_1', c_eps,
             '%.2f' % random_ac_1, '%.4f' % random_time_1,
             '%.2f' % random_ac_2, '%.4f' % random_time_2],
        ])
    for _ in range(1):
        noise_type = random.choice(noise_types)
        noise_list = ["bit_flip", "depolarizing", "phase_flip"] if noise_type == 'mixed' else []
        # noise_p = random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075])
        # eps = random.choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01])
        # eps = choice([0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.075])
        noise_p = random.choice(probs)
        print('*' * 40 + "verifying {} with {}_{}".format(model_name, noise_type, noise_p) + '*' * 40)
        final_kraus, noise_name = generating_circuit_with_specified_noise(random_circuit, random_kraus,
                                                                          noise_type, noise_list,
                                                                          kraus_file, noise_p, model_name)
        final_ac_temp, final_time_temp = verifier(final_kraus, O, data, label, c_eps, type,
                                                  GET_NEW_DATASET, 0, ADVERSARY_EXAMPLE,
                                                  digits, 'mnist')
        final_ac_1 = final_ac_temp[0] * 100
        final_ac_2 = final_ac_temp[1] * 100
        final_time_1 = final_time_temp[0]
        final_time_2 = final_time_temp[1]
        noise_ = noise_type.replace('_', '-')
        with open("./results/local_results.csv", "a+") as csvfile:
            w = csv.writer(csvfile)
            w.writerow([model_name, 'c_2 ({}_{})'.format(noise_, noise_p), c_eps,
                        '%.2f' % final_ac_1, '%.4f' % final_time_1,
                        '%.2f' % final_ac_2, '%.4f' % final_time_2])


if len(argv) > 2:
    # python evaluate.py ./model_and_data/iris.qasm ./model_and_data/iris_data.npz mixed
    # python evaluate.py ./model_and_data/tfi4.qasm ./model_and_data/tfi4_data.npz pure
    # python evaluate.py ./model_and_data/tfi8.qasm ./model_and_data/tfi8_data.npz pure
    # python evaluate.py ./model_and_data/fashion8.qasm ./model_and_data/fashion8_data.npz pure
    # python evaluate.py ./model_and_data/mnist13.qasm ./model_and_data/mnist13_data.npz pure
    evaluate()
else:
    # for example:
    # python evaluate.py 13
    digits = str(argv[1])
    evaluate_mnist(digits)
