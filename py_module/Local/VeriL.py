import cvxpy as cp
import numpy as np
import time
from scipy.optimize import minimize, NonlinearConstraint
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Agg')


def StateRobustnessVerifier(OO, data, label, e):
    dim, n = data.shape[1], data.shape[0]
    e = 1. - np.sqrt(1. - e)
    non_robust_num = 0
    new_data = []
    new_labels = []
    # non_robust_index = []
    print('Starting state robustness verifier...')
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
            new_data.append(sigma.value)
            new_labels.append(label[i])
            # non_robust_index.append(True)
        # else:
        # non_robust_index.append(False)

        print('{:d}/{:d} states checked: {:d} unrobust state'.format(i + 1, n, non_robust_num), end='\r')

    print('{:d}/{:d} states checked: {:d} unrobust state'.format(i + 1, n, non_robust_num))
    return non_robust_num, new_data, new_labels


def PureStateRobustnessVerifier(OO, data, label, e, ADVERSARY_EXAMPLE=False, digits='36'):
    dim, n = data.shape[0], data.shape[1]
    non_robust_num = 0
    C = lambda x: x.reshape((-1, 1))
    # For convenience, only find real state
    OO = np.real(OO)
    data = np.real(data)
    new_data = []
    new_labels = []
    print('Starting pure state robustness verifier...')
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
            new_data.append(res.x)
            new_labels.append(label[i])
            # correct_labels.append(1 - label[i])
            if ADVERSARY_EXAMPLE:
                # original = psi.reshape((16, 16), order='F')
                # adv_example = res.x.reshape((16, 16), order='F')
                original = psi.reshape((16, 16))
                adv_example = res.x.reshape((16, 16))
                maximum = np.maximum(original.max(), adv_example.max())
                # digits = '36'

                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(original, cmap='gray', vmin=0, vmax=maximum)
                plt.colorbar(fraction=0.045, orientation='horizontal', pad=0.05)
                plt.title('label ' + digits[1 - label[i]])
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
                plt.title('label ' + digits[label[i]])
                plt.xticks([])
                plt.yticks([])

                # plt.show()
                plt.savefig('./adversary_examples/advExample_{}_{:d}.png'.format(digits, non_robust_num),
                            bbox_inches='tight')
                plt.close()
        # else:
        #     correct_labels.append(label[i])
        print('{:d}/{:d} states checked: {:d} unrobust state'.format(i + 1, n, non_robust_num), end='\r')

    print('{:d}/{:d} states checked: {:d} unrobust state'.format(i + 1, n, non_robust_num))
    return non_robust_num, new_data, new_labels


def logistic(x):
    return 1 / (1 + np.exp(-x))


def RobustnessVerifier(E, O, data, label, e, type, GET_NEW_DATASET=False, origin_dataset_size=0):
    """

    Args:
        E:
        O:
        data:
        label:
        e:
        type:
        GET_NEW_DATASET:
            GET_NEW_DATASET == True if you want to obtain a new dataset for adversarial training.
        origin_dataset_size:
        GET_ADVERSARY_EXAMPLE:

    Returns:

    """

    time_start = time.time()
    print('=' * 45 + '\nStarting Robustness Verifier...')
    print('-' * 45 + '\nChecking {:g}-robustness'.format(e))

    NKraus, dim, n = E.shape[0], data.shape[1], data.shape[0]
    OO = np.zeros([dim, dim], dtype=complex)
    non_robust_num = np.zeros([2, ], dtype=int)
    check_time = np.zeros([2, ])

    for i in range(NKraus):
        OO += E[i, :, :].conj().T @ O @ E[i, :, :]

    ex = np.zeros((n,))
    for i in range(n):
        ex[i] = np.real(np.trace(OO @ data[i, :, :]))
        if type == 'qasm':
            ex[i] = logistic(ex[i])

    non_robust_index = (np.abs(np.sqrt(ex) - np.sqrt(1. - ex)) <= (np.sqrt(2. * e))) & ((ex > 0.5) == label)
    non_robust_num[0] = np.sum(non_robust_index)
    check_time[0] = time.time() - time_start

    non_robust_num_in_origin_dataset = np.zeros([2, ], dtype=int)
    non_robust_num_in_origin_dataset[0] = np.sum(non_robust_index[:origin_dataset_size])
    non_robust_num_in_origin_dataset[1] = 0

    non_robust_num[1] = non_robust_num[0]
    if non_robust_num[1] > 0:
        print('Filted by robust bound, {:d} states left for SDP method'.format(non_robust_num[1]))
        non_robust_num[1], new_data, new_label = StateRobustnessVerifier(OO,
                                                                         data[non_robust_index, :, :],
                                                                         label[non_robust_index],
                                                                         e)
        if GET_NEW_DATASET:
            data_ = data[:origin_dataset_size, :, :]
            label_ = label[:origin_dataset_size]
            non_robust_index_ = non_robust_index[:origin_dataset_size]
            print('origin data.shape=', data_.shape)
            print('origin label.shape', label_.shape)
            print('non_robust_index_.shape', non_robust_index_.shape)
            non_robust_num_in_origin_dataset[1], _, _ = StateRobustnessVerifier(OO,
                                                             data_[non_robust_index_, :, :],
                                                             label_[non_robust_index_],
                                                             e)
            print('non_robust_num_1 =', non_robust_num_in_origin_dataset[0])
            print('non_robust_num_2 =', non_robust_num_in_origin_dataset[1])
            new_datas = np.zeros([n + len(new_label), dim, dim], dtype=complex)
            new_labels = np.zeros([n + len(new_label)], dtype=complex)
            # new_datas[:n, :, :] = data
            # new_labels[:n] = label
            new_datas[:n, :, :] = copy.deepcopy(data)
            new_labels[:n] = copy.deepcopy(label)
            for i in range(len(new_label)):
                new_datas[n + i, :, :] = copy.deepcopy(new_data[i])
                new_labels[i] = new_label[i]
            print('new data.shape=', new_datas.shape)
            print('new label.shape=', new_labels.shape)
    else:
        print('Filted by robust bound, all states are robust')

    check_time[1] = time.time() - time_start

    robust_ac = 1. - non_robust_num / np.double(n)
    print('-' * 45 + '\nVerification over')
    print('Robust accuracy: {:.2f}%,'.format(robust_ac[1] * 100), end=' ')
    print('Verification time: {:.2f}s'.format(check_time[1]))
    print('=' * 45 + '\n')
    if GET_NEW_DATASET:
        return robust_ac, check_time, non_robust_num_in_origin_dataset, new_datas, new_labels
    else:
        return robust_ac, check_time


def PureRobustnessVerifier(E, O, data, label, e, type, GET_NEW_DATASET=False, origin_dataset_size=0,
                           GET_ADVERSARY_EXAMPLE=False, digits='36', filename=''):
    """

    Args:
        E:
        O:
        data:
        label:
        e:
        type:
        GET_NEW_DATASET:
            GET_NEW_DATASET == True if you want to obtain a new dataset for adversarial training.
        origin_dataset_size:
        GET_ADVERSARY_EXAMPLE:
        digits:
        filename:

    Returns:

    """
    time_start = time.time()
    print('=' * 45 + '\nStarting Pure Robustness Verifier...')
    print('-' * 45 + '\nChecking {:g}-robustness'.format(e))

    NKraus, dim, n = E.shape[0], data.shape[0], data.shape[1]
    OO = np.zeros([dim, dim], dtype=complex)
    non_robust_num = np.zeros([2, ], dtype=int)
    check_time = np.zeros([2, ])

    for i in range(NKraus):
        OO += E[i, :, :].conj().T @ O @ E[i, :, :]

    ex = np.zeros((n,))
    for i in range(n):
        ex[i] = np.real(data[:, i].T.conj() @ OO @ data[:, i])
        if type == 'qasm' and 'mnist' not in filename:
            ex[i] = logistic(ex[i])

    non_robust_index = (np.abs(np.sqrt(ex) - np.sqrt(1. - ex)) <= (np.sqrt(2. * e))) & ((ex > 0.5) == label)
    non_robust_num[0] = np.sum(non_robust_index)
    check_time[0] = time.time() - time_start

    non_robust_num_in_origin_dataset = np.zeros([2, ], dtype=int)
    non_robust_num_in_origin_dataset[0] = np.sum(non_robust_index[:origin_dataset_size])
    non_robust_num_in_origin_dataset[1] = 0

    non_robust_num[1] = non_robust_num[0]
    if non_robust_num[1] > 0:
        print('Filted by robust bound, {:d} states left for QCQP method'.format(non_robust_num[1]))
        non_robust_num[1], new_data, new_label = PureStateRobustnessVerifier(OO,
                                                                             data[:, non_robust_index],
                                                                             label[non_robust_index],
                                                                             e,
                                                                             GET_ADVERSARY_EXAMPLE,
                                                                             digits)
        if GET_NEW_DATASET:
            data_ = data[:, :origin_dataset_size]
            label_ = label[:origin_dataset_size]
            non_robust_index_ = non_robust_index[:origin_dataset_size]
            non_robust_num_in_origin_dataset[1], _, _ = PureStateRobustnessVerifier(OO,
                                                                 data_[:, non_robust_index_],
                                                                 label_[non_robust_index_],
                                                                 e,
                                                                 GET_ADVERSARY_EXAMPLE,
                                                                 digits)
            print('non_robust_num_2 =', non_robust_num_in_origin_dataset[1])
            new_datas = np.zeros([dim, n + len(new_label)], dtype=complex)
            new_labels = np.zeros([n + len(new_label)], dtype=complex)
            # new_datas[:, :n] = data
            # new_labels[:n] = label
            new_datas[:, :n] = copy.deepcopy(data)
            new_labels[:n] = copy.deepcopy(label)
            for i in range(len(new_label)):
                new_datas[:, n + i] = copy.deepcopy(new_data[i])
                new_labels[i] = new_label[i]
            print('origin data.shape=', data.shape)
            print('origin label.shape=', label.shape)
            print('new data.shape=', new_datas.shape)
            print('new label.shape=', new_labels.shape)
    else:
        print('Filted by robust bound, all states are robust')

    check_time[1] = time.time() - time_start

    robust_ac = 1. - non_robust_num / np.double(n)
    print('-' * 45 + '\nVerification over')
    print('Robust accuracy: {:.2f}%,'.format(robust_ac[1] * 100), end=' ')
    print('Verification time: {:.2f}s'.format(check_time[1]))
    print('=' * 45 + '\n')
    if GET_NEW_DATASET:
        return robust_ac, check_time, non_robust_num_in_origin_dataset, new_datas, new_labels
    else:
        return robust_ac, check_time
