from base64 import encode
from abc import ABC, abstractmethod
import os

project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(project_path, './'))

import numpy as np
import mindspore as ms
from mindspore.nn import Adam, Accuracy
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import Callback
from mindquantum import *
from mindquantum.algorithm import HardwareEfficientAnsatz
from mindquantum.core import RY, RZ, RX, Rxx, Ryy, Rzz
import matplotlib.pyplot as plt

from mindspore import ops, Tensor
from sklearn.model_selection import train_test_split

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)


class QCNN(object):
    '''Quantum Convolutional Neural Network Class


    '''

    def __init__(self,
                 qubits,
                 closed=True,
                 learning_rate=0.001,
                 epoch=8,
                 batch=8,
                 opt=False):
        self.circ = None
        self.encoder = None
        self.ansatz = None
        self.ham = None
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch = batch
        self.qubits = qubits
        self.closed = closed
        self.opt = opt
        X, Y = self.load_data()
        Xext, Yext = self.data_ext(X, Y, 10)
        X_train, X_test, Y_train, Y_test = train_test_split(Xext,
                                                            Yext,
                                                            test_size=0.2,
                                                            random_state=0,
                                                            shuffle=True)
        self.train_x = X_train
        self.train_y = Y_train
        self.test_x = X_test
        self.test_y = Y_test
        self.dataset = self.build_dataset(self.train_x, self.train_y,
                                          self.batch)
        self.test_dataset = self.build_dataset(self.test_x, self.test_y,
                                               self.batch)
        self.qnet = MQLayer(self.build_grad_ops())
        self.model = self.build_model()
        self.checkpoint_name = os.path.join(project_path, "model.ckpt")

    def load_data(self):
        '''Load data from local folder: ``tfi_chain/spin_systems/TFI_chain/closed``
        '''
        gamma_list = [i / 100.0 for i in range(20, 182, 2)]
        xlist = []
        ylist = []
        for i in gamma_list:
            f = '%s/tfi_chain/spin_systems/TFI_chain/closed/%s/%.2f/params.npy' % (
                data_path, self.qubits, i)
            data = np.load(f).T
            xlist.append(data.reshape(-1))
            if i > 1:
                ylist.append(1)
            else:
                ylist.append(0)
        print('Load data finished.')
        return np.array(xlist), np.array(ylist)

    def data_ext(self, X, Y, factor=10):
        '''Enhance the original data based on linear interpolation

        Args:
            X: array or list, the original data X
            Y: array or list, the original label data Y
            factor: int, data size enhancement factor, default to 10.

        Returns:
            extended data Xext, Yext
        '''
        Xext = []
        Yext = []
        gamma_list = [i / 100.0 for i in range(20, 182, 2)]
        for i in range(1, len(X)):
            dx = (X[i] - X[i - 1]) / (factor)
            dy = (gamma_list[i] - gamma_list[i - 1]) / (factor)
            for j in range(factor):
                Xext.append(X[i - 1] + j * dx)
                if gamma_list[i - 1] + j * dy > 1:
                    Yext.append(1)
                else:
                    Yext.append(0)
        Xext.append(X[-1])
        if gamma_list[i - 1] > 1:
            Yext.append(1)
        else:
            Yext.append(0)
        print('Extended Data shape: %s, %s, factor: %s' %
              (np.array(Xext).shape, np.array(Yext).shape, factor))
        return np.array(Xext), np.array(Yext)

    def build_dataset(self, x, y, batch=None, shuffle=False):
        train = ds.NumpySlicesDataset({
            "image": x,
            "label": y
        },
            shuffle=shuffle)
        if batch is not None:
            train = train.batch(batch)
        return train

    def gen_encoder(self):
        '''Generate encoder circuit:
        based on the qubits of ``qcnn``, the ``layers`` is set to ``int(qubits/2)``, each layer of encoder is construct by ``Rzz`` and ``RX`` gates.

        Returns:
            encoder circuit
        '''
        qubits = self.qubits
        encoder = Circuit()
        encoder += UN(H, qubits)
        layers = int(qubits / 2)
        for l in range(layers):
            for i in range(qubits - 1):
                encoder += Rzz(f'alpha_{l}_0').on([i, i + 1])
                encoder += RX(f'alpha_{l}_1').on(i)
            encoder += Rzz(f'alpha_{l}_0').on([0, qubits - 1])
            encoder += RX(f'alpha_{l}_1').on(qubits - 1)
        encoder = encoder.no_grad()
        return encoder

    def q_convolution(self, label, qubits):
        '''Generate 2 qubits convolution circuit: given by two qubits index, generate the convolution block which is constructed by ``RX, RY, RZ, Rxx, Ryy, Rzz`` gates.

        Args:
            label: str, name of the convolution block
            qubits: array or list, the qubits index list of the convolution block

        Returns:
            convolution circuit
        '''
        count = 0
        circ = Circuit()
        for i in range(2):
            circ += RX(f'cov_{label}_{count}').on(qubits[i])
            count += 1
            circ += RY(f'cov_{label}_{count}').on(qubits[i])
            count += 1
            circ += RZ(f'cov_{label}_{count}').on(qubits[i])
            count += 1
        circ += Rxx(f'cov_{label}_{count}').on(qubits)
        count += 1
        circ += Ryy(f'cov_{label}_{count}').on(qubits)
        count += 1
        circ += Rzz(f'cov_{label}_{count}').on(qubits)
        count += 1
        for i in range(2):
            circ += RX(f'cov_{label}_{count}').on(qubits[i])
            count += 1
            circ += RY(f'cov_{label}_{count}').on(qubits[i])
            count += 1
            circ += RZ(f'cov_{label}_{count}').on(qubits[i])
        return circ

    def q_pooling(self, label, qubits, last=False):
        '''Generate 2 qubits pooling circuit: given by two qubits index, generate the pooling block which is constructed by ``RX,RY,RZ,X`` gates.
        When optimized settings are taken (``self.opt`` is set to ``True``), the pooling circuit is optimized.

        Args:
            label: str, name of the pooling block
            qubits: array or list, the qubits index list of the pooling block
            last: bool, used to determine whether the pooling block is the last one

        Returns:
            pooling circuit
        '''
        count = 0
        circ = Circuit()
        if not self.opt:
            for i in range(2):
                circ += RX(f'p_{label}_{count}').on(qubits[i])
                count += 1
                circ += RY(f'p_{label}_{count}').on(qubits[i])
                count += 1
                circ += RZ(f'p_{label}_{count}').on(qubits[i])
                count += 1
        circ += X.on(qubits[1], qubits[0])
        if count != 0:
            count -= 1
        if (self.opt and last) or (not self.opt):
            circ += RZ(f'p_{label}_{count}').on(qubits[1]).hermitian()
            count -= 1
            circ += RY(f'p_{label}_{count}').on(qubits[1]).hermitian()
            count -= 1
            circ += RX(f'p_{label}_{count}').on(qubits[1]).hermitian()
        return circ

    def split_qlist(self, qlist):
        '''Generate the index list of convolution and pooling block: given by the qubits index list, generate the fisrt and the second qubit index for convolution and pooling block.

        Args:
            qlist: array or list, the qubits index list

        Returns:
            flist: the list of the fisrt qubit index
            slist: the list of the second qubit index
        '''
        n = len(qlist)
        flist = []
        slist = []
        if n > 1:
            for i in range(int(n / 2)):
                flist.append(qlist[2 * i])
                slist.append(qlist[2 * i + 1])
        if n % 2 == 1:
            slist.append(qlist[-1])
        return flist, slist

    def gen_qcnn_ansatz(self):
        '''Generate the ansatz circuit of qcnn: based on the ``q_convolution, q_pooling`` functions, generate ansatz circuit.

        Returns:
            circ: ansatz circuit
        '''
        qubits = self.qubits
        assert qubits % 2 == 0
        qlist = list(range(qubits))
        circ = Circuit()
        flist, slist = self.split_qlist(qlist)
        count = 0
        while len(flist) > 0:
            if len(flist) == 1 and len(slist) == 1:
                last = True
            else:
                last = False
            for i in range(len(flist)):
                q = [flist[i], slist[i]]
                circ += self.q_convolution('blo_%s' % count, q)
                count += 1
                circ += self.q_pooling('blo_%s' % count, q, last)
                count += 1
            flist, slist = self.split_qlist(slist)
        return circ

    def build_grad_ops(self):
        '''Generate the total qcnn circuit, the Hamiltonian operator and build the grad ops wrapper.

        Returns:
            grad_ops: the grad ops wrapper
        '''
        encoder = self.gen_encoder()
        ansatz = self.gen_qcnn_ansatz()
        self.encoder = encoder
        self.ansatz = ansatz
        total_circ = encoder.as_encoder() + ansatz.as_ansatz()
        self.circ = total_circ
        qubits = self.qubits
        ham = [
            Hamiltonian(QubitOperator(f'Z{i}'))
            for i in [int(qubits / 2) - 1, qubits - 1]
        ]
        print(ham)
        self.ham = ham
        sim = Simulator('mqvector', total_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(
            ham,
            total_circ,
            parallel_worker=8)
        return grad_ops

    def build_model(self):
        '''Set the loss function, optimizer and build the qcnn model.
        '''
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True,
                                                        reduction='mean')
        self.opti = ms.nn.Adam(self.qnet.trainable_params(),
                               learning_rate=self.learning_rate,
                               beta1=0.9,
                               beta2=0.99)
        self.model = Model(self.qnet,
                           self.loss,
                           self.opti,
                           metrics={'Acc': Accuracy()})
        return self.model

    def train(self, num, callbacks=None):
        '''Training the model.

        Args:
            num: int, the epoch of training
            callbacks: list, the list of callbacks
        '''
        self.model.train(num,
                         self.dataset,
                         dataset_sink_mode=False,
                         callbacks=callbacks)

    def export_trained_parameters(self):
        '''Export the parameters of model and save model.
        '''
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        '''Load model parameters by checkpoint data.
        '''
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = np.argmax(ops.Softmax()(self.model.predict(Tensor(test_x))),
                            axis=1)
        predict = predict.flatten() > 0
        return predict


class StepAcc(Callback):  # 定义一个关于每一步准确率的回调函数

    def __init__(self, model, test_loader, qnet, qubits, opt):
        self.model = model
        self.qnet = qnet
        self.qubits = qubits
        self.opt = opt
        self.test_loader = test_loader
        self.acc = []

    def step_end(self, run_context):
        """
        Record training accuracy and save model at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        self.acc.append(
            self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc'])
        if self.acc[-1] > 0.98:
            print('save model, %s' % self.acc[-1])
            ms.save_checkpoint(
                self.qnet, "%s/res/model_%.2f_%s_%s.ckpt" %
                           (data_path, self.acc[-1], self.qubits, self.opt))


class LossMonitor(Callback):

    def __init__(self, per_print_times=1):
        super(LossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError(
                "The argument 'per_print_times' must be int and >= 0, "
                "but got {}".format(per_print_times))
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self._loss = []
        self._count = 0

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(
                    loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))

        cur_step_in_epoch = (cb_params.cur_step_num -
                             1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(
                "epoch: {} step: {}. Invalid loss, terminating training.".
                format(cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and (
                cb_params.cur_step_num -
                self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            print("epoch: %s step: %s, loss is %s" %
                  (cb_params.cur_epoch_num, cur_step_in_epoch, loss),
                  flush=True)
            self._loss.append([self._count * self._per_print_times, loss])
            self._count += 1


def logistic(x):
    return 1 / (1 + np.exp(-x))


M_1 = np.kron(np.kron(np.identity(2 ** 2), Z.matrix()), I.matrix())
M_3 = np.kron(Z.matrix(), np.identity(2 ** 3))
M_4qubit = M_1 - M_3

M_7 = np.kron(M_3, np.identity(16))
M_3 = np.kron(np.identity(16), M_3)
M_8qubit = M_3 - M_7

M_5 = np.kron(np.kron(np.identity(2 ** 6), Z.matrix()), np.identity(2 ** 5))
M_11 = np.kron(Z.matrix(), np.identity(2 ** 11))
M_12qubit = M_5 - M_11

# M = M_4qubit
# measure_on = [1, 3]
# M = M_8qubit
# measure_on = [3, 7]
M = M_12qubit
measure_on = [5, 11]

if __name__ == '__main__':
    qubits = 12
    opt = True
    qc = QCNN(qubits, learning_rate=0.01, opt=opt)
    # print(qc.circ.summary())
    # print(qc.circ)
    # monitor = LossMonitor(5)
    # accu = StepAcc(qc.model, qc.test_dataset, qc.qnet, qubits, opt)
    # qc.train(1, [monitor, accu])
    #
    # correct = qc.model.eval(qc.test_dataset, dataset_sink_mode=False)
    # print(correct)

    # ******************************* veri ********************************** #
    pr_ansatz_12 = {'cov_blo_0_0': -0.15042722, 'cov_blo_0_1': -0.43680307, 'cov_blo_0_2': 0.05304768,
                    'cov_blo_0_3': -0.2434316, 'cov_blo_0_4': 0.28818572, 'cov_blo_0_5': -0.03482823,
                    'cov_blo_0_6': -0.23248455, 'cov_blo_0_7': 0.57361734, 'cov_blo_0_8': 0.18520367,
                    'cov_blo_0_9': -0.09862347, 'cov_blo_0_10': -0.41477156, 'cov_blo_0_11': -0.29930523,
                    'cov_blo_0_12': 0.22035208, 'cov_blo_0_13': -0.08405781, 'cov_blo_2_0': 0.029316617,
                    'cov_blo_2_1': -0.15558146, 'cov_blo_2_2': -0.24236706, 'cov_blo_2_3': 0.16697924,
                    'cov_blo_2_4': 0.37775415, 'cov_blo_2_5': 0.19596425, 'cov_blo_2_6': -0.38076395,
                    'cov_blo_2_7': 0.08343613, 'cov_blo_2_8': -0.39975062, 'cov_blo_2_9': 0.096727856,
                    'cov_blo_2_10': -0.27801567, 'cov_blo_2_11': 0.16271417, 'cov_blo_2_12': 0.4055084,
                    'cov_blo_2_13': 0.1280661, 'cov_blo_4_0': -0.20092995, 'cov_blo_4_1': -0.24955957,
                    'cov_blo_4_2': 0.4295829, 'cov_blo_4_3': -0.31412074, 'cov_blo_4_4': 0.26089266,
                    'cov_blo_4_5': -0.013178526, 'cov_blo_4_6': -0.07367884, 'cov_blo_4_7': -0.4659945,
                    'cov_blo_4_8': 0.029424997, 'cov_blo_4_9': -0.2816276, 'cov_blo_4_10': -0.28922218,
                    'cov_blo_4_11': -0.40386617, 'cov_blo_4_12': 0.18718874, 'cov_blo_4_13': 0.24377424,
                    'cov_blo_6_0': 0.13783337, 'cov_blo_6_1': 0.03287855, 'cov_blo_6_2': 0.25713795,
                    'cov_blo_6_3': 0.07254067, 'cov_blo_6_4': 0.026284974, 'cov_blo_6_5': -0.18876633,
                    'cov_blo_6_6': -0.7319484, 'cov_blo_6_7': 0.48486868, 'cov_blo_6_8': -0.20478953,
                    'cov_blo_6_9': 0.1979761, 'cov_blo_6_10': 0.17328013, 'cov_blo_6_11': 0.00577665,
                    'cov_blo_6_12': -0.036944047, 'cov_blo_6_13': 0.22892761, 'cov_blo_8_0': 0.1099596,
                    'cov_blo_8_1': 0.010468608, 'cov_blo_8_2': -0.51835686, 'cov_blo_8_3': 0.020024493,
                    'cov_blo_8_4': 0.02510388, 'cov_blo_8_5': -0.13738975, 'cov_blo_8_6': -0.3614961,
                    'cov_blo_8_7': -0.15370193, 'cov_blo_8_8': -0.16535039, 'cov_blo_8_9': 0.10374704,
                    'cov_blo_8_10': -0.09571105, 'cov_blo_8_11': 0.016222533, 'cov_blo_8_12': 0.02424852,
                    'cov_blo_8_13': 0.2575953, 'cov_blo_10_0': 0.097759664, 'cov_blo_10_1': 0.014977735,
                    'cov_blo_10_2': -0.0024596127, 'cov_blo_10_3': 0.092140265, 'cov_blo_10_4': 0.0069678505,
                    'cov_blo_10_5': -0.07845256, 'cov_blo_10_6': 0.13774812, 'cov_blo_10_7': 0.24853095,
                    'cov_blo_10_8': -0.07790844, 'cov_blo_10_9': 0.14708884, 'cov_blo_10_10': 0.014956684,
                    'cov_blo_10_11': 0.14924003, 'cov_blo_10_12': -0.041742504, 'cov_blo_10_13': 0.0733705,
                    'cov_blo_12_0': -0.3097367, 'cov_blo_12_1': 0.35862547, 'cov_blo_12_2': 0.32214707,
                    'cov_blo_12_3': 0.19091074, 'cov_blo_12_4': -0.08113095, 'cov_blo_12_5': -0.39455837,
                    'cov_blo_12_6': -0.36635333, 'cov_blo_12_7': 0.26142922, 'cov_blo_12_8': -0.076060735,
                    'cov_blo_12_9': -0.3661348, 'cov_blo_12_10': 0.4119638, 'cov_blo_12_11': 0.2133773,
                    'cov_blo_12_12': -0.21993807, 'cov_blo_12_13': -0.43092567, 'cov_blo_14_0': -0.40644363,
                    'cov_blo_14_1': 0.20856218, 'cov_blo_14_2': -0.05114832, 'cov_blo_14_3': -0.012711095,
                    'cov_blo_14_4': -0.19801465, 'cov_blo_14_5': -0.1320104, 'cov_blo_14_6': -0.16097833,
                    'cov_blo_14_7': -0.35502583, 'cov_blo_14_8': -0.4296668, 'cov_blo_14_9': -0.36745444,
                    'cov_blo_14_10': 0.28099197, 'cov_blo_14_11': -0.3594022, 'cov_blo_14_12': -0.41954866,
                    'cov_blo_14_13': 0.56075686, 'cov_blo_16_0': 0.02454553, 'cov_blo_16_1': -0.0058299582,
                    'cov_blo_16_2': -0.02295467, 'cov_blo_16_3': 0.17885043, 'cov_blo_16_4': -0.1360573,
                    'cov_blo_16_5': 0.1358746, 'cov_blo_16_6': 0.45673874, 'cov_blo_16_7': -0.075213015,
                    'cov_blo_16_8': -0.24026449, 'cov_blo_16_9': 0.07821033, 'cov_blo_16_10': -0.028373396,
                    'cov_blo_16_11': 0.1261016, 'cov_blo_16_12': -0.1584042, 'cov_blo_16_13': 0.28719524,
                    'cov_blo_18_0': 0.22023377, 'cov_blo_18_1': -0.45139334, 'cov_blo_18_2': -0.3490481,
                    'cov_blo_18_3': -0.22756767, 'cov_blo_18_4': -0.3661236, 'cov_blo_18_5': 0.19633491,
                    'cov_blo_18_6': -0.09476363, 'cov_blo_18_7': -0.16988963, 'cov_blo_18_8': -0.20090313,
                    'cov_blo_18_9': 0.30097216, 'cov_blo_18_10': -0.47463277, 'cov_blo_18_11': -0.35855556,
                    'cov_blo_18_12': -0.3423118, 'cov_blo_18_13': 0.1838289, 'cov_blo_20_0': -0.35577527,
                    'cov_blo_20_1': -0.315509, 'cov_blo_20_2': -0.026193338, 'cov_blo_20_3': 0.085723996,
                    'cov_blo_20_4': -0.40525138, 'cov_blo_20_5': -0.027251724, 'cov_blo_20_6': -0.38403553,
                    'cov_blo_20_7': 0.41416264, 'cov_blo_20_8': -0.42315608, 'cov_blo_20_9': -0.4725312,
                    'cov_blo_20_10': -0.27793688, 'cov_blo_20_11': -0.456492, 'cov_blo_20_12': -0.4091867,
                    'cov_blo_20_13': 0.08159973, 'p_blo_21_0': 0.47905305, 'p_blo_21_-1': 0.3761727,
                    'p_blo_21_-2': 0.5056397}
    # pr_ansatz = dict(zip(qc.ansatz.params_name, qc.qnet.weight.asnumpy()))  # 获取线路参数
    pr_ansatz = pr_ansatz_12
    print('pr_ansatz = ', pr_ansatz)

    ansatz_ = qc.ansatz.apply_value(pr_ansatz)
    U = ansatz_.matrix()

    ansatz_ += Measure('Z{}'.format(measure_on[0])).on(measure_on[0])
    ansatz_ += Measure('Z{}'.format(measure_on[1])).on(measure_on[1])
    ansatz_.svg().to_file('./TFIchain{}_model.svg'.format(qubits))

    # print(ansatz_)
    ansatz_str = OpenQASM().to_string(ansatz_)
    f = open('./TFIchain{}.qasm'.format(qubits), 'w')
    f.write(ansatz_str)
    f.close()
    ansatz_ = ansatz_.remove_measure()

    predict_by_expectation = []
    predict_by_M = []
    predict_by_prob = []
    # num = 20
    # print(qc.train_x[:num].shape)
    # print(qc.train_x[:num])
    data = []
    for sample in qc.train_x:
        print(sample)
        pr_encoder = dict(zip(qc.encoder.params_name, sample))  # 获取线路参数
        # print('pr_encoder = ', pr_encoder)
        encoder_ = qc.encoder.apply_value(pr_encoder)
        # circuit_ = encoder_ + ansatz_

        # sim_ = Simulator('mqmatrix', circuit_.n_qubits)
        # sim_.apply_circuit(circuit_)

        # m1 = np.real(sim_.get_expectation(qc.ham[0]))
        # m2 = np.real(sim_.get_expectation(qc.ham[1]))
        # # print('m1 = {}, m2 = {}'.format(m1, m2))
        # if m1 > m2:
        #     predict_by_expectation.append(0)
        # else:
        #     predict_by_expectation.append(1)

        # rho = encoder_.get_qs(backend='mqmatrix')
        state = encoder_.get_qs(backend='mqvector')
        data.append(state)
    #     m = np.real(np.trace(U.conj().T @ M @ U @ rho))
    #     if m > 0:
    #         predict_by_M.append(0)
    #     else:
    #         predict_by_M.append(1)
    #
    #     p0 = logistic(m)
    #     p1 = 1 - p0
    #     if p0 > p1:
    #         predict_by_prob.append(0)
    #     else:
    #         predict_by_prob.append(1)
    #
    # for i in range(len(predict_by_prob)):
    #     if (predict_by_prob[i] != predict_by_expectation[i] or
    #             predict_by_prob[i] != predict_by_M[i] or
    #             predict_by_expectation[i] != predict_by_M[i]):
    #         print("{}th error".format(i+1))
    # break

    # print('predict_by_expe: ', np.array(predict_by_expectation))
    # print('   predict_by_M: ', np.array(predict_by_prob))
    # print('predict_by_prob: ', np.array(predict_by_prob))
    # print('actual  y_train: ', qc.train_y)

    data = np.array(data).T
    label = [1 - i for i in qc.train_y]
    label = np.array(label)
    print(M.shape)
    print(data.shape)
    print(label.shape)
    np.savez('./tfi{}_data.npz'.format(qubits), O=M, data=data, label=label)

    # plt.figure()
    # plt.plot(accu.acc)
    # plt.title('Statistics of accuracy', fontsize=20)
    # plt.xlabel('Steps', fontsize=20)
    # plt.ylabel('Accuracy', fontsize=20)
    # plt.grid(ls=":", c='b')
    # plt.savefig('acc_%s_%s.png' % (qubits, opt), format='png')
    # plt.figure()
    # monitor._loss = np.array(monitor._loss)
    # plt.plot(monitor._loss[:, 0], monitor._loss[:, 1])
    # plt.title('Statistics of loss', fontsize=20)
    # plt.xlabel('Steps', fontsize=20)
    # plt.ylabel('Loss', fontsize=20)
    # plt.grid(ls=":", c='b')
    # plt.savefig('loss_%s_%s.png' % (qubits, opt), format='png')
