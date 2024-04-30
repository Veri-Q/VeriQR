import mindspore as ms
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

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)


def mat_m(qubit_num):
    M_0 = np.kron(I.matrix(), Z.matrix())
    M_1 = np.kron(Z.matrix(), I.matrix())
    for i in range(qubit_num - 2):
        M_0 = np.kron(M_0, I.matrix())
        M_1 = np.kron(M_1, I.matrix())

    return M_0 - M_1


QUBIT_NUM = 8
M = mat_m(QUBIT_NUM)


def logistic(x):
    return 1 / (1 + np.exp(-x))


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


def amplitude_param(pixels):
    param_rd = []
    _, parameterResolver = amplitude_encoder(pixels, QUBIT_NUM)
    for _, param in parameterResolver.items():
        param_rd.append(param)
    param_rd = np.array(param_rd)
    return param_rd


class ForwardAndLoss(ms.nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(ForwardAndLoss, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        output = self.backbone(data)
        return self.loss_fn(output, label)

    def backbone_network(self):
        return self.backbone


class TrainOneStep(ms.nn.TrainOneStepCell):
    def __init__(self, network, optimizer):
        super(TrainOneStep, self).__init__(network, optimizer)
        self.grad = ms.ops.GradOperation(get_by_list=True)

    def construct(self, data, label):
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)


class EpochLoss(ms.nn.Metric):
    def __init__(self):
        super(EpochLoss, self).__init__()
        self.clear()

    def clear(self):
        self.loss = 0
        self.counter = 0

    def update(self, *loss):
        loss = loss[0].asnumpy()
        self.loss += loss
        self.counter += 1

    def eval(self):
        return self.loss / self.counter


class EpochAcc(ms.nn.Metric):
    def __init__(self):
        super(EpochAcc, self).__init__()
        self.clear()

    def clear(self):
        self.correct_num = 0
        self.total_num = 0

    def update(self, *inputs):
        y_output = inputs[0].asnumpy()
        y = inputs[1].asnumpy()
        y_pred = np.zeros_like(y)
        for i in range(y_pred.shape[0]):
            yi = y_output[i]
            if yi[0] >= yi[1]:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        self.correct_num += np.sum(y == y_pred)
        self.total_num += y.shape[0]

    def eval(self):
        return self.correct_num / self.total_num


class Main():
    def __init__(self, setname, Ent_name):
        super().__init__()
        self.dataset_file = './Dataset/'
        self.setname = setname
        self.Ent = Ent_name
        self.train_loader, self.test_loader = self.data_preporcess(
            TRAIN_SET_NUM, TEST_SET_NUM)

    def data_preporcess(self, trainset_num, testset_num):
        train_num = trainset_num
        test_num = testset_num
        '''
        原论文给的数据集.mat文件保存格式不同, 需要分别用h5py和scipy打开
        '''
        if self.setname == 'FashionMNIST':
            self.dataset = h5py.File(self.dataset_file +
                                     '/FashionMNIST_1_2_wk.mat')
            train_data = np.transpose(self.dataset['x_train'])
            train_label = np.transpose(self.dataset['y_train'])
            test_data = np.transpose(self.dataset['x_test'])
            test_label = np.transpose(self.dataset['y_test'])

            train_pixels = np.array(train_data[:, :train_num].tolist())[:, :, 0].transpose()  # [:,:,0]取实部
            test_pixels = np.array(test_data[:, :test_num].tolist())[:, :, 0].transpose()
            train_index = train_label[:train_num, 0].astype(int)  # 0: 靴子,  1: T恤
            test_index = test_label[:test_num, 0].astype(int)

        elif self.setname == 'MNIST':
            self.dataset = scipy.io.loadmat(self.dataset_file +
                                            '/MNIST_1_9_wk.mat')
            train_data = self.dataset['x_train']
            train_label = self.dataset['y_train']
            test_data = self.dataset['x_test']
            test_label = self.dataset['y_test']

            train_pixels = np.real(train_data[:, :train_num]).transpose()
            test_pixels = np.real(test_data[:, :test_num]).transpose()
            train_index = train_label[:train_num, 0].astype(int)  # 0->0, 1->9
            test_index = test_label[:test_num, 0].astype(int)

        # 将幅度转为编码线路参数，幅度shape(256,)，参数shape(255,)
        train_param = np.array([amplitude_param(i) for i in train_pixels])
        test_param = np.array([amplitude_param(i) for i in test_pixels])

        train_loader = ms.dataset.NumpySlicesDataset({
            'features': train_param,
            'labels': train_index
        }, shuffle=True).batch(BATCH_SIZE)
        test_loader = ms.dataset.NumpySlicesDataset({
            'features': test_param,
            'labels': test_index
        }).batch(BATCH_SIZE)
        return train_loader, test_loader

    def training(self):
        '''搭建量子线路'''
        encoder = amplitude_encoder([0], QUBIT_NUM)[0].as_encoder()
        ansatz = Classifying_circuit(QUBIT_NUM, BLOCK_NUM,
                                     self.Ent).as_ansatz()
        circ = encoder.as_encoder() + ansatz
        meas = [
            Hamiltonian(QubitOperator(f'Z{i}'))
            for i in [QUBIT_NUM - 2, QUBIT_NUM - 1]
        ]  # 测量最后两个比特
        sim = Simulator('mqvector', circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(meas,
                                                 circ,
                                                 parallel_worker=WORKER)
        self.Qnet = MQLayer(grad_ops)
        '''构建训练模型'''
        loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True,
                                                   reduction='mean')
        opt = ms.nn.Adam(self.Qnet.trainable_params(), learning_rate=LR)
        net_with_loss = ForwardAndLoss(self.Qnet, loss)
        train_one_step = TrainOneStep(net_with_loss, opt)
        '''定义评价指标'''
        acc_epoch = EpochAcc()
        loss_epoch = EpochLoss()
        '''训练并保存模型'''
        train_loss_epoch = []
        train_acc_epoch = []
        test_loss_epoch = []
        test_acc_epoch = []

        for epoch in range(STEP_NUM):
            loss_epoch.clear()
            acc_epoch.clear()

            for data in self.train_loader:
                train_one_step(data[0], data[1])  # 执行训练，并更新权重, data[0]参数，data[1]为标签
                loss = net_with_loss(data[0], data[1])
                loss_epoch.update(loss)
            train_loss = loss_epoch.eval()
            train_loss_epoch.append(train_loss)

            # training accuracy
            for data in self.train_loader:
                logits = self.Qnet(data[0])  # 向前传播得到预测值
                acc_epoch.update(logits, data[1])  # 计算预测准确率
            train_acc = acc_epoch.eval()
            train_acc_epoch.append(train_acc)

            # testing loss
            for data in self.test_loader:
                loss = net_with_loss(data[0], data[1])  # 计算损失值
                loss_epoch.update(loss)
            test_loss = loss_epoch.eval()
            test_loss_epoch.append(test_loss)

            # testing accuracy
            for data in self.test_loader:
                logits = self.Qnet(data[0])
                acc_epoch.update(logits, data[1])
            test_acc = acc_epoch.eval()
            test_acc_epoch.append(test_acc)

            print(
                f"epoch: {epoch + 1}, training loss: {train_loss}, training acc: {train_acc}, testing loss: {test_loss}, testing acc: {test_acc}"
            )

        def get_trained_ansatz():
            pr_ansatz = dict(zip(ansatz.params_name, self.Qnet.weight.asnumpy()))  # 获取线路参数
            print('pr_ansatz = ', pr_ansatz)

            ansatz_ = ansatz.apply_value(pr_ansatz)
            U = ansatz_.matrix()
            print(ansatz_)

            ansatz_ += Measure('Z{}'.format(QUBIT_NUM - 2)).on(QUBIT_NUM - 2)
            ansatz_ += Measure('Z{}'.format(QUBIT_NUM - 1)).on(QUBIT_NUM - 1)
            ansatz_.svg().to_file("../../figures/FashionMNIST_{}_model.svg".format(self.Ent))

            # print(ansatz_)
            ansatz_str = OpenQASM().to_string(ansatz_)
            f = open('../../model_and_data/FashionMNIST_{}.qasm'.format(self.Ent), 'w')
            f.write(ansatz_str)
            f.close()
            return ansatz_, U

        ansatz_, U = get_trained_ansatz()
        ansatz_ = ansatz_.remove_measure()

        def veri():
            data = []
            predict_by_measure = []
            predict_by_prob = []
            predict_by_expectation = []
            X_train = [data[0] for data in self.train_loader]
            y_train = [data[1] for data in self.train_loader]
            count_error = 0
            for i in range(int(TRAIN_SET_NUM / BATCH_SIZE)):
                X_train_ = X_train[i]  # 一个batch, 即100个样本点
                y_train_ = y_train[i]
                for sample in X_train_:
                    pr_encoder = dict(zip(encoder.params_name, sample.tolist()))  # 获取线路参数
                    # print('pr_encoder = ', pr_encoder)

                    encoder_ = encoder.apply_value(pr_encoder)
                    rho = encoder_.get_qs(backend='mqmatrix')
                    state = encoder_.get_qs(backend='mqvector')
                    data.append(state)

                    # circuit_ = encoder_ + ansatz_
                    # sim_ = Simulator('mqvector', QUBIT_NUM)
                    # sim_.apply_circuit(circuit_)
                    # m1 = np.real(sim_.get_expectation(hams[0]))
                    # m2 = np.real(sim_.get_expectation(hams[1]))
                    # if m1 > m2:
                    #     predict_by_expectation.append(0)
                    # else:
                    #     predict_by_expectation.append(1)

                    m = np.real(np.trace(U.conj().T @ M @ U @ rho))
                    if m > 0:
                        predict_by_measure.append(0)
                    else:
                        predict_by_measure.append(1)

                    p0 = logistic(m)
                    p1 = 1 - p0
                    if p0 > p1:
                        predict_by_prob.append(0)
                    else:
                        predict_by_prob.append(1)

                for j in range(len(y_train_)):
                    if predict_by_measure[j + i * BATCH_SIZE] != y_train_[j]:
                        print("the {}th sample prediction error".format(j + 1 + i * BATCH_SIZE))
                        print('(measure) predict: ', predict_by_measure[j + i * BATCH_SIZE])
                        print('actual: ', y_train_[j])
                        count_error += 1
                    if predict_by_prob[j + i * BATCH_SIZE] != y_train_[j]:
                        print("the {}th sample prediction error".format(j + 1 + i * BATCH_SIZE))
                        print('(prob) predict: ', predict_by_prob[j + i * BATCH_SIZE])
                        print('actual: ', y_train_[j])
                        print()

                # print('predict_by_expectation: ', predict_by_measure)
                # print('actual y_train: ', y_test_)

            data = np.array(data).T
            label = [1 - i for i in predict_by_measure]
            label = np.array(label)
            print('count_error:', count_error)
            print('acc:', 1 - count_error/len(label))
            print(U.shape)
            print(M.shape)
            print(data.shape)
            print(label.shape)
            np.savez('../../model_and_data/FashionMNIST_{}_data.npz'.format(self.Ent), O=M, data=data, label=label)

        veri()


# if __name__ == '__main__':
BLOCK_NUM = 2
BATCH_SIZE = 100
LR = 0.01
STEP_NUM = 30  # 60
WORKER = 5
TRAIN_SET_NUM = 1000
TEST_SET_NUM = 200

main = Main('FashionMNIST', 'Ent1')
main.training()
