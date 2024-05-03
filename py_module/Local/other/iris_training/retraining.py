import mindspore as ms
from mindquantum import *
import mindspore.nn as nn
from mindspore.nn import Adam, TrainOneStepCell, LossBase
from mindspore import ops, Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations
import mindspore.numpy as msNumpy
import numpy as np

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)  # 设置生成随机数的种子


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
        # out = self.abs(logistic(logits) / 2 + 0.5 - label)
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


# training


def retaining(data_file):
    DATA = np.load('../../model_and_data/newdata_for_AT/{}'.format(data_file))
    X_train = Tensor(DATA['data'], ms.complex128)
    y_train = Tensor(DATA['label'], ms.int32)

    ansatz = HardwareEfficientAnsatz(4, single_rot_gate_seq=[RY], entangle_gate=X, depth=3).circuit
    circuit = ansatz.as_ansatz()

    # 搭建量子神经网络
    sim = Simulator('mqmatrix', circuit.n_qubits)
    hams = Hamiltonian(QubitOperator('Z2'))
    grad_ops = sim.get_expectation_with_grad(hams, circuit)
    myloss = MyLoss()
    qnet = AnsatzOnlyLayer(grad_ops)
    net = MyWithLossCell(qnet, myloss)
    opti = Adam(qnet.trainable_params(), learning_rate=0.1)
    train_one_step = TrainOneStepCell(net, opti)

    def training(X_train: Tensor, y_train: Tensor, epochs: int):
        print('training...')
        res = 0.
        for epoch in range(epochs):
            for i in range(len(X_train)):
                res = train_one_step(X_train[i], y_train[i])
            # print('epoch {}: {}'.format(epoch, res))
            validating(X_train, y_train, qnet)

    def validating(x: Tensor, y: Tensor, qnet):
        loss = 0
        acc = 0
        for i in range(len(x)):
            expectation = qnet(x[i])
            loss += float(myloss(expectation, y[i]))
            predict = 1 if expectation > 0 else 0
            acc += int(predict == y[i])

        loss /= len(x)
        acc /= len(x)
        print('loss:', loss)
        print('acc:', acc)
        return loss, acc

    training(X_train, y_train, 20)

    def get_trained_ansatz():
        pr_ansatz = dict(zip(ansatz.params_name, qnet.weight.asnumpy()))  # 获取线路参数
        # print('pr_ansatz = ', pr_ansatz)

        ansatz_ = ansatz.apply_value(pr_ansatz)
        U = ansatz_.matrix()
        # print(ansatz_)

        file_name = data_file[:-4]
        ansatz_ += Measure('Z2').on(2)
        ansatz_ += Measure('Z3').on(3)
        # ansatz_ += Measure('Z{}'.format(QUBIT_NUM - 1)).on(QUBIT_NUM - 1)
        ansatz_.svg().to_file('../../model_and_data/newmodel_by_AT/{}.svg'.format(file_name))

        # print(ansatz_)
        ansatz_str = OpenQASM().to_string(ansatz_)
        f = open('../../model_and_data/newmodel_by_AT/{}.qasm'.format(file_name), 'w')
        f.write(ansatz_str)
        f.close()

        np.savez('../../model_and_data/newmodel_by_AT/{}.npz'.format(file_name), kraus=np.array([U]))
        # return ansatz_, U

    get_trained_ansatz()


retaining('iris_c0_by_0.01.npz')
retaining('iris_c0_by_0.05.npz')
retaining('iris_c1_by_0.01.npz')
retaining('iris_c1_by_0.05.npz')
retaining('iris_c2_BitFlip_0.005_by_0.01.npz')
retaining('iris_c2_BitFlip_0.005_by_0.05.npz')
retaining('iris_c2_Depolarizing_0.0005_by_0.01.npz')
retaining('iris_c2_Depolarizing_0.0005_by_0.05.npz')
retaining('iris_c2_PhaseFlip_0.005_by_0.01.npz')
retaining('iris_c2_PhaseFlip_0.005_by_0.05.npz')
retaining('iris_c2_mixed_BitFlip_Depolarizing_PhaseFlip_0.0005_by_0.01.npz')
retaining('iris_c2_mixed_BitFlip_Depolarizing_PhaseFlip_0.0005_by_0.05.npz')
