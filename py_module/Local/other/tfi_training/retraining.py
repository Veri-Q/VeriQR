import os

project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(project_path, './'))

import mindspore as ms
from mindspore.nn import Adam, Accuracy, TrainOneStepCell, LossBase
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import Callback
from mindquantum import *
from mindquantum.core import RY, RZ, RX, Rxx, Ryy, Rzz
from mindspore import ops, Tensor
from sklearn.model_selection import train_test_split
import mindspore.nn as nn
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

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


class AnsatzOnlyOps(nn.Cell):
    def __init__(self, expectation_with_grad: GradOpsWrapper):
        """Initialize a MQAnsatzOnlyOps object."""
        super().__init__()
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = operations.Shape()
        self.g = None

    def construct(self, arg, init_state: ms.Tensor):  # 此处新增参数 init_state 用以传入初始态
        self.expectation_with_grad.sim.set_qs(init_state.asnumpy())  # 此处将模拟器初态设置为 init_state
        fval, g_ans = self.expectation_with_grad(arg.asnumpy())
        self.g = np.real(g_ans[0])
        return ms.Tensor(np.real(fval[0]), dtype=ms.float32)

    def bprop(self, arg, out, tmp, dout):  # pylint: disable=unused-argument
        """Implement the bprop function."""
        dout = dout.asnumpy()
        grad = dout @ self.g
        return ms.Tensor(grad, dtype=ms.float32)


class AnsatzOnlyLayer(nn.Cell):
    def __init__(self, expectation_with_grad, weight='normal'):
        """Initialize a MQAnsatzOnlyLayer object."""
        super().__init__()
        self.evolution = AnsatzOnlyOps(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get f{weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')

    def construct(self, init_state: ms.Tensor):  # 新增参数 init_state
        """Construct a MQAnsatzOnlyLayer node."""
        return self.evolution(self.weight, init_state)


def logistic(x: Tensor):
    return 1 / (1 + ms.numpy.exp(-x))


class MyLoss(LossBase):
    def __init__(self, reduction='mean'):
        super(MyLoss, self).__init__(reduction)
        self.abs = ops.Abs()

    def construct(self, logits, label):
        # out = self.abs(logits / 2 + 0.5 - label)
        out = self.abs(logistic(logits) - label)
        return self.get_loss(out)


class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, x, label):
        out = self._backbone(x)
        return self._loss_fn(out, label)


def retaining(data_file, qubits_num):
    DATA = np.load('../../model_and_data/newdata_for_AT/{}'.format(data_file))
    X_train = Tensor(DATA['data'], ms.complex128)
    y_train = Tensor(DATA['label'], ms.int32)
    print('数据集信息：')
    print(X_train.shape)  # 打印训练集中样本的数据类型
    print(y_train.shape)
    print('')
    TRAIN_SET_NUM = X_train.shape[1]

    ansatz = Classifying_circuit(QUBIT_NUM, BLOCK_NUM, 'Ent1').as_ansatz()
    circuit = ansatz.as_ansatz()

    # 搭建量子神经网络
    sim = Simulator('mqvector', circuit.n_qubits)  # pure states
    ham_map = {
        4: QubitOperator('Z1'),
        8: QubitOperator('Z3'),
        12: QubitOperator('Z5'),
    }
    hams = Hamiltonian(ham_map[qubits_num])
    opt = True
    qc = QCNN(qubits_num, learning_rate=0.01, opt=opt)
    print(qc.circ.summary())
    print(qc.circ)
    monitor = LossMonitor(5)
    accu = StepAcc(qc.model, qc.test_dataset, qc.qnet, qubits, opt)
    qc.train(1, [monitor, accu])

    correct = qc.model.eval(qc.test_dataset, dataset_sink_mode=False)
    print(correct)

    grad_ops = sim.get_expectation_with_grad(hams, circuit)
    myloss = MyLoss()
    qnet = AnsatzOnlyLayer(grad_ops)
    net = MyWithLossCell(qnet, myloss)
    opti = Adam(qnet.trainable_params(), learning_rate=LR)
    train_one_step = TrainOneStepCell(net, opti)

    def training(datas: Tensor, labels: Tensor, epochs: int):
        print('training...')
        res = 0.
        for epoch in range(epochs):
            for i in range(TRAIN_SET_NUM):
                res = train_one_step(datas[:, i], labels[i])
            # print('epoch {}: {}'.format(epoch, res))
            validating(datas, labels, qnet)

    def validating(x: Tensor, y: Tensor, qnet):
        loss = 0
        acc = 0
        for i in range(TRAIN_SET_NUM):
            expectation = qnet(x[:, i])
            loss += float(myloss(expectation, y[i]))
            predict = 1 if expectation > 0 else 0
            acc += int(predict == y[i])

        loss /= TRAIN_SET_NUM
        acc /= TRAIN_SET_NUM
        print('loss:', loss)
        print('acc:', acc)
        return loss, acc

    training(X_train, y_train, STEP_NUM)

    def get_trained_ansatz():
        pr_ansatz = dict(zip(ansatz.params_name, qnet.weight.asnumpy()))  # 获取线路参数
        # print('pr_ansatz = ', pr_ansatz)

        ansatz_ = ansatz.apply_value(pr_ansatz)
        U = ansatz_.matrix()
        # print(ansatz_)

        file_name = data_file[:-4]
        ansatz_ += Measure('Z{}'.format(QUBIT_NUM - 2)).on(QUBIT_NUM - 2)
        ansatz_ += Measure('Z{}'.format(QUBIT_NUM - 1)).on(QUBIT_NUM - 1)
        ansatz_.svg().to_file('../../model_and_data/newmodel_by_AT/{}.svg'.format(file_name))

        # print(ansatz_)
        ansatz_str = OpenQASM().to_string(ansatz_)
        f = open('../../model_and_data/newmodel_by_AT/{}.qasm'.format(file_name), 'w')
        f.write(ansatz_str)
        f.close()

        np.savez('../../model_and_data/newmodel_by_AT/{}.npz'.format(file_name), kraus=np.array([U]))
        # return ansatz_, U

    get_trained_ansatz()


retaining('tfi4_c0_by_0.05.npz')
retaining('tfi4_c0_by_0.1.npz')
