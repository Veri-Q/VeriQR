import cvxpy as cp
import numpy as np
import time
from scipy.optimize import minimize, NonlinearConstraint
from mindquantum.core.gates import KrausChannel
from mindquantum.core.circuit import Circuit

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


def StateRobustnessVerifier(OO, data, label, e):
    dim, n = data.shape[1], data.shape[0]
    non_robust_num = 0
    e = 1. - np.sqrt(1. - e)

    print('=' * 35 + '\nStarting state robustness verifier\n' + '-' * 35)
    for i in range(n):
        rho = data[i, :, :]
        # For convenience, only find real entries state
        sigma = cp.Variable((dim, dim), PSD=True)
        X = cp.Variable((dim, dim), complex=True)
        Y = cp.bmat([[rho, X], [X.H, sigma]])

        cons = [sigma >> 0.,
                cp.trace(sigma) == 1.,
                Y >> 0.]
        if label[i] == 0:
            cons += [cp.real(cp.trace((OO / dim) @ sigma)) >= (0.5 / dim)]
        else:
            cons += [cp.real(cp.trace((OO / dim) @ sigma)) <= (0.5 / dim)]

        obj = cp.Minimize(1 - cp.real(cp.trace(X)))

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.SCS)
        delta = 1 - (1. - prob.value) / np.trace(sigma.value)
        if delta < e:
            non_robust_num += 1

        print('{:d}/{:d} states checked: {:d} unrobust state'.format(i + 1, n, non_robust_num), end='\r')

    print('{:d}/{:d} states checked: {:d} unrobust state'.format(i + 1, n, non_robust_num))
    print('=' * 35)
    return non_robust_num


def PureStateRobustnessVerifier(OO, data, label, e, ADVERSARY_EXAMPLE=False):
    dim, n = data.shape[0], data.shape[1]
    non_robust_num = 0
    C = lambda x: x.reshape((-1, 1))
    # For convenience, only find real state
    OO = np.real(OO)
    data = np.real(data)

    print('=' * 35 + '\nStarting pure state robustness verifier\n' + '-' * 35)
    for i in range(n):
        psi = C(data[:, i])
        A = psi @ psi.conj().T
        obj = lambda phi: 1. - (C(phi).conj().T @ A @ C(phi))[0, 0]
        obj_J = lambda phi: -2. * (A @ C(phi))[:, 0]
        obj_H = lambda phi: -2. * A

        if label[i] == 0:
            cons_f = lambda phi: [0.5 - (C(phi).conj().T @ OO @ C(phi))[0, 0],
                                  (C(phi).conj().T @ C(phi))[0, 0] - 1.,
                                  1. - (C(phi).conj().T @ C(phi))[0, 0]]
            cons_J = lambda phi: [-2. * (OO @ C(phi))[:, 0], 2. * C(phi)[:, 0], -2. * C(phi)[:, 0]]
            cons_H = lambda phi, v: 2. * (v[0] * -OO + v[1] * np.eye(dim) - v[2] * np.eye(dim))
        else:
            cons_f = lambda phi: [(C(phi).conj().T @ OO @ C(phi))[0, 0] - 0.5,
                                  (C(phi).conj().T @ C(phi))[0, 0] - 1.,
                                  1. - (C(phi).conj().T @ C(phi))[0, 0]]
            cons_J = lambda phi: [2. * (OO @ C(phi))[:, 0], 2. * C(phi)[:, 0], -2. * C(phi)[:, 0]]
            cons_H = lambda phi, v: 2. * (v[0] * OO + v[1] * np.eye(dim) - v[2] * np.eye(dim))

        cons = NonlinearConstraint(cons_f, -np.inf, 0, jac=cons_J, hess=cons_H)
        res = minimize(obj, psi[:, 0], method='trust-constr', jac=obj_J, hess=obj_H,
                       constraints=[cons])

        delta = 1. - (1. - obj(res.x)) / np.dot(res.x, res.x)
        if delta < e:
            non_robust_num += 1
            if ADVERSARY_EXAMPLE:
                original = psi.reshape((16, 16), order='F')
                adv_example = res.x.reshape((16, 16), order='F')
                maximum = np.maximum(original.max(), adv_example.max())

                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(original, cmap='gray', vmin=0, vmax=maximum)
                plt.colorbar(fraction=0.045, orientation='horizontal', pad=0.05)
                plt.title('label ' + '63'[label[i]])
                plt.xticks([])
                plt.yticks([])

                plt.subplot(1, 3, 2)
                plt.imshow(1e4 * (adv_example - original), cmap='gray')
                plt.colorbar(fraction=0.045, orientation='horizontal', pad=0.05)
                plt.text(11, 22, 'x 1e-4')
                plt.xticks([])
                plt.yticks([])
                plt.text(-2.6, 8, '+')
                plt.text(16.6, 8, '=')

                plt.subplot(1, 3, 3)
                plt.imshow(adv_example, cmap='gray', vmin=0, vmax=maximum)
                plt.colorbar(fraction=0.045, orientation='horizontal', pad=0.05)
                plt.title('label ' + '36'[label[i]])
                plt.xticks([])
                plt.yticks([])

                #                plt.show()
                plt.savefig('./adversary_examples/adversary_example_{:d}.png'.format(non_robust_num),
                            bbox_inches='tight')
                plt.close()

        print('{:d}/{:d} states checked: {:d} unrobust state'.format(i + 1, n, non_robust_num), end='\r')

    print('{:d}/{:d} states checked: {:d} unrobust state'.format(i + 1, n, non_robust_num))
    print('=' * 35)
    return non_robust_num


def logistic(x):
    return 1 / (1 + np.exp(-x))


def RobustnessVerifier(E, O, data, label, e, type):
    time_start = time.time()
    print('=' * 45 + '\nStarting Robustness Verifier\n' + '-' * 45)
    print('Checking {:g}-robustness\n'.format(e) + '-' * 40)
    NKraus, dim, n = E.shape[0], data.shape[1], data.shape[0]
    OO = np.zeros([dim, dim], dtype=complex)
    non_robust_num = np.zeros([2, ], dtype=int)
    check_time = np.zeros([2, ])

    sum_E = np.zeros([dim, dim], dtype=complex)

    if type == 'qasm':
        for i in range(NKraus):
            OO += E[i, :, :].conj().T @ O @ E[i, :, :]
            sum_E += E[i, :, :].conj().T @ E[i, :, :]
    else:
        for i in range(NKraus):
            # print(E[i, :, :].shape)
            # print(E[i, :, :].conj().T @ O @ E[i, :, :])
            OO += E[i, :, :].conj().T @ O @ E[i, :, :]
            sum_E += E[i, :, :].conj().T @ E[i, :, :]

    ex = np.zeros((n,))
    # predict = []
    for i in range(n):
        ex[i] = np.real(np.trace(OO @ data[i, :, :]))

        if type == 'qasm':
            ex[i] = logistic(ex[i])
    #     if ex[i] > 1-ex[i]:
    #         predict.append(0)
    #     else:
    #         predict.append(1)
    #
    # print('predict: ', predict)
    # print(ex)

    non_robust_index = (np.abs(np.sqrt(ex) - np.sqrt(1. - ex)) <= (np.sqrt(2. * e))) & ((ex > 0.5) == label)
    non_robust_num[0] = np.sum(non_robust_index)
    time_end = time.time()
    check_time[0] = time_end - time_start
    non_robust_num[1] = non_robust_num[0]
    if non_robust_num[1] > 0:
        print('Filted by robust bound, {:d} states left for SDP method'.format(non_robust_num[1]))
        non_robust_num[1] = StateRobustnessVerifier(OO,
                                                    data[non_robust_index, :, :],
                                                    label[non_robust_index],
                                                    e)
    else:
        print('Filted by robust bound, all states are robust')

    time_end = time.time()
    check_time[1] = time_end - time_start

    robust_ac = 1. - non_robust_num / np.double(n)
    print('Verification over\n' + '-' * 40)
    print('Robust accuracy: {:.2f}%,'.format(robust_ac[1] * 100), end=' ')
    print('Verification time: {:.2f}s'.format(check_time[1]))
    print('=' * 45)

    return robust_ac, check_time


def PureRobustnessVerifier(E, O, data, label, e, type, ADVERSARY_EXAMPLE=False):
    time_start = time.time()
    print('=' * 45 + '\nStarting Pure Robustness Verifier\n' + '-' * 45)
    print('Checking {:g}-robustness\n'.format(e) + '-' * 40)
    NKraus, dim, n = E.shape[0], data.shape[0], data.shape[1]
    OO = np.zeros([dim, dim], dtype=complex)
    non_robust_num = np.zeros([2, ], dtype=int)
    check_time = np.zeros([2, ])

    for i in range(NKraus):
        OO += E[i, :, :].conj().T @ O @ E[i, :, :]

    ex = np.zeros((n,))
    for i in range(n):
        ex[i] = np.real(data[:, i].T.conj() @ OO @ data[:, i])

    non_robust_index = (np.abs(np.sqrt(ex) - np.sqrt(1. - ex)) <= (np.sqrt(2. * e))) & ((ex > 0.5) == label)
    non_robust_num[0] = np.sum(non_robust_index)
    time_end = time.time()
    check_time[0] = time_end - time_start
    non_robust_num[1] = non_robust_num[0]
    if non_robust_num[1] > 0:
        print('Filted by robust bound, {:d} states left for QCQP method'.format(non_robust_num[1]))
        non_robust_num[1] = PureStateRobustnessVerifier(OO,
                                                        data[:, non_robust_index],
                                                        label[non_robust_index],
                                                        e,
                                                        ADVERSARY_EXAMPLE)
    else:
        print('Filted by robust bound, all states are robust')

    time_end = time.time()
    check_time[1] = time_end - time_start

    robust_ac = 1. - non_robust_num / np.double(n)
    print('Verification over\n' + '-' * 40)
    print('Robust accuracy: {:.2f}%,'.format(robust_ac[1] * 100), end=' ')
    print('Verification time: {:.2f}s'.format(check_time[1]))
    print('=' * 45)

    return robust_ac, check_time
