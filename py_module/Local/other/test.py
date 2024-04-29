import numpy as np
from mindquantum.core.gates import H, X, RZ, RY, I, Z, DepolarizingChannel, BitFlipChannel, PhaseFlipChannel

# import mindquantum.core.gates as gate

import numpy as np
from numpy import load
import random

data = load("../model_and_data/iris_data.npz")
print(data["data"].shape)
# np.savez('./kraus/kraus_1qubit.npz', kraus=kraus)

# I = gate.I.matrix()
# Z = gate.Z.matrix()


# print(I)
# print(Z)
# mat_0 = np.array([[1. + 0.j, 0. + 0.j],
#                   [0. + 0.j, 0. + 0.j]])
#
# mat_1 = np.array([[0. + 0.j, 0. + 0.j],
#                   [0. + 0.j, 1. + 0.j]])
#
# # phi = np.array([i for i in range(1, 17)])
# # print(phi)
# P2_0 = np.kron(np.kron(np.kron(I, mat_0), I), I)
# P2_1 = np.kron(np.kron(np.kron(I, mat_1), I), I)
#
# s = P2_0.conj().T @ P2_0 + P2_1.conj().T @ P2_1
# # print(s)
#
# sum = P2_0 + P2_1
#
# for i in range(len(s)):
#     if s[i][i] != 1.0:
#         print("wrong")
#
# for i in range(len(sum)):
#     if sum[i][i] != 1.0:
#         print("wrong")


# print("outcome =")
# print(phi @ M_2 @ phi.T)
#
# M_2 = np.kron(np.kron(np.kron(I, Z), I), I)
# M_3 = np.kron(np.kron(np.kron(Z, I), I), I)
# M = M_2-M_3
# print(M @ M.conj().T)
# assert np.all(np.isclose(M, M))

# print(M_3)
#
# M_0 = np.kron(np.kron(np.kron(I, I), I), Z)
# M_1 = np.kron(np.kron(np.kron(I, I), Z), I)
# print("M_0 + M_1 + M_2 + M_3 =")
# print(M_0 + M_1 + M_2 + M_3)
#
# m0 = M_0.conj().T @ M_0
# m1 = M_1.conj().T @ M_1
# m2 = M_2.conj().T @ M_2
# m3 = M_3.conj().T @ M_3
# # print("m2 =")
# # print(m2)
# # print("m3 =")
# # print(m3)
# print("m0 + m1 + m2 + m3 =")
# print(m0 + m1 + m2 + m3)

# def mat_m(qubit_num):
# M_0 = np.kron(I.matrix(), Z.matrix())
# M_1 = np.kron(Z.matrix(), I.matrix())
# for i in range(qubit_num - 2):
#     M_0 = np.kron(M_0, I.matrix())
#     M_1 = np.kron(M_1, I.matrix())
#
# return M_0 - M_1

noise_ops = ["phase_flip", "depolarizing", "bit_flip"]
noise_op_map = {
    "bit_flip": BitFlipChannel,
    "depolarizing": DepolarizingChannel,
    "phase_flip": PhaseFlipChannel,
}


def qubit_add_noise():
    data_file = "../model_and_data/qubit_cav.npz"  # 6 qubit
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


# print(data.shape)
# print(data)
# for rho in data:
#     print(np.real(np.trace(rho @ rho)))


# label = [1-i for i in label]
# print(label)
# np.savez('../model_and_data/FashionMNIST_data.npz', O=O, data=data, label=label)

def mat_m(qubit_num):
    M_0 = np.kron(I.matrix(), Z.matrix())
    M_1 = np.kron(Z.matrix(), I.matrix())
    for i in range(qubit_num - 2):
        M_0 = np.kron(M_0, I.matrix())
        M_1 = np.kron(M_1, I.matrix())

    return M_0 - M_1


# M = mat_m(8)
# print(M.shape)


# def lossfunc():
#     # 原函数
#     def argminf(params):
#         # r = ((x1 + x2 - 4) ** 2 + (2 * x1 + 3 * x2 - 7) ** 2 + (4 * x1 + x2 - 9) ** 2) * 0.5
#         r = 0
#         for i in range(len(X_train)):
#             pr_encoder = dict(zip(encoder.params_name, X_train[i]))  # 获取线路参数
#             # print('pr_encoder = ', pr_encoder)
#             encoder_ = encoder.apply_value(pr_encoder)
#
#             pr = dict(zip(ansatz.params_name, params))
#             ansatz_ = ansatz.apply_value(pr)
#
#             # circuit_ = encoder_ + ansatz_
#
#             rho = encoder_.get_qs(backend='mqmatrix')
#             # print('current state: ', rho)
#
#             U = ansatz_.matrix()
#             m1_ = np.real(np.trace(U.conj().T @ M_2 @ U @ rho))  # 第1个测量值
#             m2_ = np.real(np.trace(U.conj().T @ M_3 @ U @ rho))  # 第2个测量值
#             print('m1_=', m1_)
#             print('m2_=', m2_)
#             y = sign()
#             if y_train[i] == 0:
#                 if m1_ > m2_:  # 期望训练样本中标签为“0”的样本的第1个测量值更大
#                     r -= 1
#                 else:
#                     r += 1
#             else:
#                 if m2_ > m1_:  # 期望标签为“1”的样本的第2个测量值更大
#                     r -= 1
#                 else:
#                     r += 1
#         print("loss = ", r)
#         return r
#
#     # 全量计算一阶偏导的值
#     def deriv_x(x1, x2):
#         r1 = (x1 + x2 - 4) + 2 * (2 * x1 + 3 * x2 - 7) * 2 + (4 * x1 + x2 - 9) * 4
#         r2 = (x1 + x2 - 4) + 2 * (2 * x1 + 3 * x2 - 7) * 3 + (4 * x1 + x2 - 9)
#         return r1, r2
#
#     # 梯度下降算法
#     def gradient_decs(n):
#         alpha = 0.1  # 学习率
#         params = np.zeros(len(encoder.params_name), dtype=float)  # 初始值
#         deriv_of_params = deriv_x(params)
#         y1 = argminf(params)
#         for i in range(n):
#             for i in range(len(params)):
#                 params[i] = params[i] - alpha * deriv_of_params[i]
#
#             y2 = argminf(params)
#             if y1 - y2 < 1e-6:
#                 return params, y2
#             if y2 < y1:
#                 y1 = y2
#         return params, y2
#
#     # 迭代1000次结果
#     gradient_decs(1000)


def printdata():
    # np.set_printoptions(threshold=np.inf)

    # data_file = "binary_cav.npz"         # 1 qubit
    # data_file = "mnist_cav.npz"          # 8 qubit
    # data_file = "excitation_cav.npz"     # 6 qubit
    data_file = "../model_and_data/tfi8_data.npz"  # 6 qubit
    DATA = np.load(data_file)
    data = DATA['data']
    # for rho in data:
    #     print(np.real(np.trace(rho @ rho)))
    print(data.shape)

# printdata()
