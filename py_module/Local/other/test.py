from common_interface import *
import numpy as np
import random


def qubit_add_noise():
    data_file = '../model_and_data/qubit_cav.npz' # 6 qubit
    DATA = np.load(data_file)
    kraus = DATA['kraus']
    O = DATA['O']
    data = DATA['data']
    label = DATA['label']

    p = float(round(random.uniform(0, 0.001), 5))
    E = noise_op_map[random.choice(noise_ops)](p).matrix()
    kraus_ = []
    for e in E:
        kraus_.append(e @ kraus[0])
    kraus_ = np.array(kraus_)
    print(kraus_.shape)
    np.savez('../model_and_data/qubitRandom_cav.npz', O=O, data=data, label=label, kraus=kraus_)

    p = 0.001
    new_kraus_1 = []
    E = DepolarizingChannel(p).matrix()
    print(E)
    for k in kraus_:
        for e in E:
            new_kraus_1.append(e @ k)

    new_kraus_1 = np.array(new_kraus_1)
    print(new_kraus_1.shape)
    np.savez('../model_and_data/qubitDepolarizing{}_cav.npz'.format(p), O=O, data=data, label=label, kraus=new_kraus_1)

    p = 0.005
    new_kraus_2 = []
    E = DepolarizingChannel(p).matrix()
    print(E)
    for k in kraus_:
        for e in E:
            new_kraus_2.append(e @ k)

    new_kraus_2 = np.array(new_kraus_2)
    print(new_kraus_2.shape)
    np.savez('../model_and_data/qubitDepolarizing{}_cav.npz'.format(p), O=O, data=data, label=label, kraus=new_kraus_2)


qubit_add_noise()


def printdata(data_file):
    # np.set_printoptions(threshold=np.inf)

    # data_file = "binary_cav.npz"         # 1 qubit
    # data_file = "mnist_cav.npz"          # 8 qubit
    # data_file = "excitation_cav.npz"     # 6 qubit
    DATA = np.load(data_file)
    data = DATA['data']
    # for rho in data:
    #     print(np.real(np.trace(rho @ rho)))
    print(data.shape)

# printdata('../model_and_data/tfi8_data.npz')
