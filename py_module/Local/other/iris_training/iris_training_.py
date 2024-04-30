import numpy as np
from sklearn import datasets  # 导入datasets模块，用于加载鸢尾花的数据集
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, X, RZ, RY, I, Z, Measure
from mindquantum.algorithm.nisq import HardwareEfficientAnsatz
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.io import OpenQASM
import mindspore as ms
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Adam  # 用于定义损失函数和优化参数
from mindspore.train import Accuracy, Model, LossMonitor  # Accuracy模块用于评估预测准确率
from mindspore.dataset import NumpySlicesDataset  # NumpySlicesDataset模块用于创建模型可以识别的数据集
from mindspore import ops  # 导入ops模块
from math import log

from mindspore import Tensor


iris_dataset = datasets.load_iris()  # 加载鸢尾花的数据集

print(iris_dataset.data.shape)
print(iris_dataset.feature_names)  # 样本的特征名称
print(iris_dataset.target_names)   # 样本包含的亚属名称
print(iris_dataset.target)         # 样本的标签
print(iris_dataset.target.shape)

x = iris_dataset.data[:100, :].astype(np.float32)  # 选取iris_dataset的data的前100个数据，将其数据类型转换为float32，并储存在x中
X_feature_names = iris_dataset.feature_names  # 将iris_dataset的特征名称储存在X_feature_names中
y = iris_dataset.target[:100].astype(int)  # 选取iris_dataset的target的前100个数据，将其数据类型转换为int，并储存在y中
y_target_names = iris_dataset.target_names[:2]  # 选取iris_dataset的target_names的前2个数据，并储存在y_target_names中

print('x.shape', x.shape)  # 打印样本的数据维度
print(X_feature_names)  # 打印样本的特征名称
# print(y)  # 打印样本的标签的数组
print('y.shape', y.shape)  # 打印样本的标签的数据维度
print(y_target_names)  # 打印样本包含的亚属名称

feature_name = {0: 'sepal length', 1: 'sepal width', 2: 'petal length', 3: 'petal width'}  # 将不同的特征名称分别标记为0,1,2,3
axes = plt.figure(figsize=(23, 23)).subplots(4, 4)  # 画出一个大小为23*23的图，包含4*4=16个子图

colormap = {0: 'r', 1: 'g'}  # 将标签为0的样本设为红色，标签为1的样本设为绿色
cvalue = [colormap[i] for i in y]  # 将100个样本对应的标签设置相应的颜色

for i in range(4):
    for j in range(4):
        if i != j:
            ax = axes[i][j]  # 在[i][j]的子图上开始画图
            ax.scatter(x[:, i], x[:, j], c=cvalue)  # 画出第[i]个特征和第[j]个特征组成的散点图
            ax.set_xlabel(feature_name[i], fontsize=22)  # 设置X轴的名称为第[i]个特征名称，字体大小为22
            ax.set_ylabel(feature_name[j], fontsize=22)  # 设置Y轴的名称为第[j]个特征名称，字体大小为22
# plt.show()  # 渲染图像，即呈现图像

# 数据预处理
alpha = x[:, :3] * x[:, 1:]  # 每一个样本中，利用相邻两个特征值计算出一个参数，即每一个样本会多出3个参数（因为有4个特征值），并储存在alpha中
x = np.append(x, alpha, axis=1)  # 在axis=1的维度上，将alpha的数据值添加到x的特征值中

print(x.shape)  # 打印此时X的样本的数据维度

circ = 'c0'
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)  #
# # 将数据集划分为训练集和测试集
# print(type(X_train))
# print(type(y_train))
DATA = np.load('../model_and_data/iris_newdata_{}.npz'.format(circ))
# X_train = Tensor(DATA['data'][:60], ms.complex128)
# y_train = Tensor(DATA['label'][:60], ms.int32)
# X_test = Tensor(DATA['data'][:20], ms.complex128)
# y_test = Tensor(DATA['label'][:20], ms.int32)
X_train = np.array(DATA['data'][:60]).astype(np.float32)
y_train = np.array(DATA['label'][:60]).astype(np.int32)
X_test = np.array(DATA['data'][:20]).astype(np.float32)
y_test = np.array(DATA['label'][:20]).astype(np.int32)
# X_train = np.array(DATA['data'][:60]).astype(np.float32)
# y_train = DATA['label'][:60]
# X_test = DATA['data'][:20]
# y_test = DATA['label'][:20]
# for i in range(10):
#     print(X_train[i])
#     print(y_train[i])
print(type(X_train))
print(type(y_train))

print(X_train.shape)  # 打印训练集中样本的数据类型
print(X_test.shape)

# 搭建Encoder
encoder = Circuit()  # 初始化量子线路

encoder += UN(H, 4)  # H门作用在每1位量子比特
for i in range(4):  # i = 0, 1, 2, 3
    encoder += RZ(f'alpha{i}').on(i)  # RZ(alpha_i)门作用在第i位量子比特
for j in range(3):  # j = 0, 1, 2
    encoder += X.on(j + 1, j)  # X门作用在第j+1位量子比特，受第j位量子比特控制
    encoder += RZ(f'alpha{j + 4}').on(j + 1)  # RZ(alpha_{j+4})门作用在第0位量子比特
    encoder += X.on(j + 1, j)  # X门作用在第j+1位量子比特，受第j位量子比特控制

encoder = encoder.no_grad()  # Encoder作为整个量子神经网络的第一层，不用对编码线路中的梯度求导数，因此加入no_grad()
encoder.summary()  # 总结Encoder
encoder.svg()

# 搭建Ansatz
ansatz = HardwareEfficientAnsatz(4, single_rot_gate_seq=[RY], entangle_gate=X, depth=3).circuit
ansatz.summary()
ansatz.svg()

circuit = encoder.as_encoder() + ansatz.as_ansatz()  # 完整的量子线路由Encoder和Ansatz组成
circuit.summary()
circuit.svg()

# 构建哈密顿量
hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [2, 3]]  # 分别对第2位和第3位量子比特执行泡利Z算符测量，且将系数都设为1，构建对应的哈密顿量
for h in hams:
    print(h)

M_2 = np.kron(np.kron(np.kron(I.matrix(), Z.matrix()), I.matrix()), I.matrix())
M_3 = np.kron(np.kron(np.kron(Z.matrix(), I.matrix()), I.matrix()), I.matrix())
M = M_2 - M_3


def sign(x_):
    return 0 if x_ > 0 else 1


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def custom_training():
    def argminf(params):
        pr = dict(zip(ansatz.params_name, params))
        # print(pr)
        ansatz_ = ansatz.apply_value(pr)
        U = ansatz_.matrix()
        # print(U)

        loss = 0
        f = np.zeros(len(X_train), dtype=float)
        y = np.zeros(len(X_train), dtype=float)
        l = np.zeros(len(X_train), dtype=float)
        for i in range(len(X_train)):
            pr_encoder = dict(zip(encoder.params_name, X_train[i]))  # 获取线路参数
            encoder_ = encoder.apply_value(pr_encoder)

            rho = encoder_.get_qs(backend='mqmatrix')
            # print('current state: ', rho)

            m1_ = np.real(np.trace(U.conj().T @ M_2 @ U @ rho))  # 第1个测量值
            m2_ = np.real(np.trace(U.conj().T @ M_3 @ U @ rho))  # 第2个测量值
            # print('m1_=', m1_)
            # print('m2_=', m2_)
            f[i] = logistic(m2_ - m1_)
            y[i] = y_train[i]
            l[i] = -(y[i] * log(f[i]) + (1 - y[i]) * log(1 - f[i]))
            loss += l[i]

        loss /= len(X_train)

        deriv_w = np.zeros(len(params), dtype=float)
        for j in range(len(deriv_w)):
            for i in range(len(X_train)):
                deriv_w[j] += -(y[i] - f[i]) * X_train[i][j % 7]

        return loss, deriv_w

    def gradient_decs(n):
        alpha = 0.1  # 学习率
        params = np.zeros(len(ansatz.params_name), dtype=float)  # 初始值
        # deriv_of_params = deriv_x(params)
        y1, deriv_of_params = argminf(params)
        for i in range(n):
            for j in range(len(params)):
                params[j] = params[j] - alpha * deriv_of_params[j]

            y2, deriv_of_params = argminf(params)

            if 0 < y1 - y2 < 1e-6:
                return params, y2

            y1 = min(y1, y2)
            print("loss = ", y1)
        return params, y2

    # 迭代100次
    opt_params, loss_ = gradient_decs(1000)

    print("optimal parameters = ", opt_params)
    print("optimal loss = ", loss_)

    pr_ansatz = dict(zip(ansatz.params_name, opt_params))  # 获取线路参数
    print('pr_ansatz = ', pr_ansatz)

    ansatz_ = ansatz.apply_value(pr_ansatz)
    U = ansatz_.matrix()

    # 检验准确率
    acc = 0
    predict_by_expectation = []
    for i in range(len(X_test)):
        pr_encoder = dict(zip(encoder.params_name, X_test[i]))  # 获取线路参数
        # print('pr_encoder = ', pr_encoder)

        encoder_ = encoder.apply_value(pr_encoder)
        circuit_ = encoder_ + ansatz_

        rho = encoder_.get_qs(backend='mqmatrix')
        # print('current state: ', rho)

        sim_ = Simulator('mqvector', circuit.n_qubits)
        sim_.apply_circuit(circuit_)

        # sim_.sampling(circuit_, shots=1000).svg().to_file("./sample_2{}.svg".format(i))  # 运行线路1000次并打印结果

        m1 = np.real(sim_.get_expectation(hams[0]))
        m2 = np.real(sim_.get_expectation(hams[1]))
        print('m1 = ', m1)
        print('m2 = ', m2)

        m1_ = np.real(np.trace(U.conj().T @ M_2 @ U @ rho))
        m2_ = np.real(np.trace(U.conj().T @ M_3 @ U @ rho))
        print('m1_ = ', m1_)
        print('m2_ = ', m2_)

        if m1_ > m2_:
            predict_by_expectation.append(0)
            if y_test[i] == 0:
                acc += 1
        else:
            predict_by_expectation.append(1)
            if y_test[i] == 1:
                acc += 1

    acc /= len(X_test)
    print('predict_by_expectation = ', predict_by_expectation)
    print('actual y_test: ', y_test)
    print('accuracy = ', acc)


# custom_training()

def prepare_data():
    data = []
    for sample in X_train:
        pr = dict(zip(encoder.params_name, sample))  # 获取线路参数
        # print(pr)

        encoder_ = encoder.apply_value(pr)

        rho = encoder_.get_qs(backend='mqmatrix')
        data.append(rho)
        print(rho.shape)

    data = np.array(data)
    print(data.shape)
    return data


def train():
    # 搭建量子神经网络
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    ms.set_seed(1)  # 设置生成随机数的种子
    sim = Simulator('mqvector', circuit.n_qubits)
    print(sim.get_qs())
    grad_ops = sim.get_expectation_with_grad(hams, circuit, parallel_worker=5)
    QuantumNet = MQLayer(grad_ops)
    # print(QuantumNet)

    # 定义损失函数，sparse=True表示指定标签使用稀疏格式，reduction='mean'表示损失函数的降维方法为求平均值
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # 通过Adam优化器优化Ansatz中的参数，需要优化的是QuantumNet中可训练的参数，学习率设为0.1
    opti = Adam(QuantumNet.trainable_params(), learning_rate=0.1)

    # 建立模型
    model = Model(QuantumNet, loss, opti, metrics={'Acc': Accuracy()})

    # 创建训练样本的数据集，shuffle=False表示不打乱数据，batch(5)表示训练集每批次样本点有5个
    train_loader = NumpySlicesDataset({'features': X_train, 'labels': y_train}, shuffle=False).batch(5)
    # # 创建测试样本的数据集，batch(5)表示测试集每批次样本点有5个
    test_loader = NumpySlicesDataset({'features': X_test, 'labels': y_test}).batch(5)

    class StepAcc(ms.Callback):  # 定义一个关于每一步准确率的回调函数
        def __init__(self, model, test_loader):
            self.model = model
            self.test_loader = test_loader
            self.acc = []

        def on_train_step_end(self, run_context):
            self.acc.append(self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc'])

    monitor = LossMonitor(16)  # 监控训练中的损失，每16步打印一次损失值

    acc = StepAcc(model, test_loader)  # 使用建立的模型和测试样本计算预测的准确率

    model.train(20, train_loader, callbacks=[monitor, acc], dataset_sink_mode=False)  # 将上述建立好的模型训练20次

    pr_ansatz = dict(zip(ansatz.params_name, QuantumNet.weight.asnumpy()))  # 获取线路参数
    print('pr_ansatz = ', pr_ansatz)

    ansatz_ = ansatz.apply_value(pr_ansatz)
    U = ansatz_.matrix()

    ansatz_ += Measure('Z2').on(2)
    ansatz_ += Measure('Z3').on(3)
    # ansatz_.svg().to_file("../figures/iris_model.svg")

    # print(ansatz_)
    ansatz_str = OpenQASM().to_string(ansatz_)
    f = open('../model_and_data/iris_{}.qasm'.format(circ), 'w')
    f.write(ansatz_str)
    f.close()

    plt.plot(acc.acc)
    plt.title('Statistics of accuracy', fontsize=20)
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)

    ansatz_ = ansatz_.remove_measure()

    predict_by_expectation = []
    predict_by_prob = []

    def veri():
        for sample in X_train:
            pr_encoder = dict(zip(encoder.params_name, sample))  # 获取线路参数
            # print('pr_encoder = ', pr_encoder)

            encoder_ = encoder.apply_value(pr_encoder)
            circuit_ = encoder_ + ansatz_

            sim_ = Simulator('mqvector', circuit.n_qubits)
            sim_.apply_circuit(circuit_)
            m1 = np.real(sim_.get_expectation(hams[0]))
            m2 = np.real(sim_.get_expectation(hams[1]))
            print('m1 = ', m1)
            print('m2 = ', m2)

            rho = encoder_.get_qs(backend='mqmatrix')
            # m1_ = np.real(np.trace(U.conj().T @ M_2 @ U @ rho))
            # m2_ = np.real(np.trace(U.conj().T @ M_3 @ U @ rho))
            # if m1_ > m2_:
            #     predict_by_expectation.append(0)
            # else:
            #     predict_by_expectation.append(1)

            m = np.real(np.trace(U.conj().T @ M @ U @ rho))

            if m > 0:
                predict_by_expectation.append(0)
            else:
                predict_by_expectation.append(1)

            p0 = logistic(m)
            p1 = 1 - p0
            if p0 > p1:
                predict_by_prob.append(0)
            else:
                predict_by_prob.append(1)

        for i in range(len(predict_by_prob)):
            if predict_by_prob[i] != predict_by_expectation[i]:
                print("error")
                break

        print('predict_by_prob: ', predict_by_prob)
        print('predict_by_expectation: ', predict_by_expectation)
        print('actual y_train: ', y_train)

    veri()

    #  预测
    predict = np.argmax(ops.Softmax()(model.predict(ms.Tensor(X_test))), axis=1)  # 使用建立的模型和测试样本，得到测试样本预测的分类
    correct = model.eval(test_loader, dataset_sink_mode=False)  # 计算测试样本应用训练好的模型的预测准确率

    print("预测分类结果：", predict)  # 对于测试样本，打印预测分类结果
    print("实际类别：", y_test)  # 对于测试样本，打印实际分类结果

    print(correct)

    data = prepare_data()
    print(predict_by_expectation)
    label = [1 - i for i in predict_by_expectation]
    label = np.array(label)
    print(label)

    # print(kraus.shape)
    print(M.shape)
    print(data.shape)
    print(label.shape)
    np.savez('../model_and_data/iris_data_{}.npz'.format(circ), O=M, data=data, label=label)


train()
