import mindspore as ms
from mindquantum import GradOpsWrapper
from mindquantum.framework import MQLayer
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.algorithm.library import amplitude_encoder
from mindquantum.simulator import Simulator
from mindquantum.core.gates import Measure, I, Z
import mindquantum.core.gates as Gate
from mindquantum.io import OpenQASM
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io
import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import Adam, TrainOneStepCell, LossBase
from mindspore import ops, Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations
import os

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)


def Classifying_circuit(qubit_num, block_num, Ent_circ):
    num = qubit_num
    depth = block_num
    circ = Circuit()
    for i in range(depth):
        circ = Para_circuit(circ, num)
        if Ent_circ == 'Ent1':
            circ = Ent1_circuit(circ, num)
        elif Ent_circ == 'Ent2':
            circ = Ent2_circuit(circ, num)
        elif Ent_circ == 'Ent3':
            circ = Ent3_circuit(circ, num)
    return circ


def Ent1_circuit(circuit, qubit_num):
    # 原文定义的Ent1纠缠层
    for i in range(0, qubit_num - 1, 2):
        circuit += Gate.Z.on(i + 1, i)
    for i in range(1, qubit_num - 2, 2):
        circuit += Gate.Z.on(i + 1, i)
    # 在MNIST分类中，此纠缠层表现更好
    # for i in range(0,qubit_num-1,1):
    #     circuit += Gate.Z.on(i+1,i)
    return circuit


def Ent2_circuit(circuit, qubit_num):
    # 原文定义的Ent2纠缠层
    for i in range(0, qubit_num - 1, 2):
        circuit += Gate.X.on(i + 1, i)
    for i in range(1, qubit_num - 2, 2):
        circuit += Gate.X.on(i + 1, i)
    # 在MNIST分类中，此纠缠层表现更好
    # for i in range(0,qubit_num-1,1):
    #     circuit += Gate.X.on(i+1,i)
    return circuit


def Ent3_circuit(circuit, qubit_num):
    circuit = Ent2_circuit(circuit, qubit_num)
    circuit = Ent2_circuit(circuit, qubit_num)
    return circuit


def Para_circuit(circuit, qubit_num):
    for i in range(qubit_num):
        # 原文中定义的参数层
        circuit += Gate.RX(f'Xtheta{i}').on(i)
        circuit += Gate.RZ(f'Ztheta{i}').on(i)
        circuit += Gate.RX(f'Xtheta2{i}').on(i)
        # 在MNIST分类中，此参数层表现更好
        # circuit += Gate.RY(f'Ytheta{i}').on(i)
    return circuit


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


QUBIT_NUM = 8
BLOCK_NUM = 2
LR = 0.01
STEP_NUM = 30  # 60

# DATA = np.load('../../model_and_data/fashion8_data.npz')
# data_file = 'fashion8_c0_by_0.001.npz'
# data_file = 'fashion8_c1_by_0.001'
# data_file = 'fashion8_c2_BitFlip_0.001_by_0.001'
# data_file = 'fashion8_c2_mixed_BitFlip_Depolarizing_PhaseFlip_0.001_by_0.001'
# args = file_name.split('_')
# model_name = args[0]
# circ_type = args[1]
# c_eps = args[len(args)-1]
# if circ_type == 'c2':
#     noise_type = args[2]
model_file = 'fashion8_c0'


# model_file = 'fashion8_c1'
# model_file = 'fashion8_c2_BitFlip'
# model_file = 'fashion8_c2_mixed_BitFlip_Depolarizing_PhaseFlip_0.001'

def retaining(data_file):
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
    hams = Hamiltonian(QubitOperator('Z6'))
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


model_name = 'fashion8'
data_path = '../../model_and_data/newdata_for_AT/'
data_list = os.listdir(data_path)
data_list = sorted(data_list, key=lambda x: os.path.getmtime(os.path.join(data_path, x)))
for file_name in data_list:
    if not file_name.startswith(model_name) or os.path.splitext(file_name)[-1] != '.npz':
        continue

    retaining(file_name)

# retaining('fashion8_c0_by_0.001.npz')
# retaining('fashion8_c0_by_0.003.npz')
# retaining('fashion8_c1_by_0.001.npz')
# retaining('fashion8_c1_by_0.003.npz')
# retaining('fashion8_c2_BitFlip_0.001_by_0.001.npz')
# retaining('fashion8_c2_BitFlip_0.001_by_0.003.npz')
# retaining('fashion8_c2_Depolarizing_0.005_by_0.001.npz')
# retaining('fashion8_c2_Depolarizing_0.005_by_0.003.npz')
# retaining('fashion8_c2_PhaseFlip_0.01_by_0.001.npz')
# retaining('fashion8_c2_PhaseFlip_0.01_by_0.003.npz')
# retaining('fashion8_c2_mixed_BitFlip_Depolarizing_PhaseFlip_0.001_by_0.001.npz')
# retaining('fashion8_c2_mixed_BitFlip_Depolarizing_PhaseFlip_0.001_by_0.003.npz')
