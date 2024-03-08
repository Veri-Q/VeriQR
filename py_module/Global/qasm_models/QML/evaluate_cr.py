import sympy
import cirq
import numpy as np
import time
import sys
import os
import pandas as pd
import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from random import choice, uniform
from numpy import load

from tensorflow.keras.callbacks import Callback

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X, Z, Y, Rxx, Power, BitFlipChannel, DepolarizingChannel, PhaseFlipChannel, Measure

noise_op_cirq = {
    "phase_flip": cirq.phase_flip,
    "depolarizing": cirq.depolarize,
    "bit_flip": cirq.bit_flip,
    "mixed": cirq.depolarize
}

noise_op_mq = {
    "phase_flip": PhaseFlipChannel,
    "depolarizing": DepolarizingChannel,
    "bit_flip": BitFlipChannel,
    "mixed": DepolarizingChannel
}

noise_ops = ["phase_flip", "depolarizing", "bit_flip"]

RANDOM = True
file_name = ''
noise_list = []
kraus_file = None
arg_num = len(sys.argv)
if arg_num <= 1:  # random noise
    noise_type = choice(noise_ops)
    noisy_p = float(round(uniform(0, 0.2), 5))  # 随机数的精度round(数值，精度)
    print("add {} with probability {}".format(noise_type, noisy_p))
    file_name = "cr_{}_{}".format(noise_type, str(noisy_p))
    # model_path = "./saved_models/cr_{}_{}".format(noise_type, str(noisy_p))
else:  # specified noise
    RANDOM = False
    noise_type = str(sys.argv[1])
    noisy_p = float(sys.argv[arg_num - 1])

    if noise_type == 'mixed':
        noise_list = [i for i in sys.argv[2: arg_num - 1]]
        noise_list_ = [noise_op_mq[i].__name__ for i in noise_list]
        noise_list_ = [i[0: i.index("Channel")] for i in noise_list_]
        print("noise_list: ", noise_list)
        file_name = "cr_mixed_{}_{}".format('_'.join(noise_list_), str(noisy_p))
        # model_path = "./saved_models/cr_mixed_{}_{}".format('_'.join(noise_list_), str(noisy_p))
    elif noise_type == 'custom':
        kraus_file = sys.argv[2]
        file_name = "cr_custom_{}_{}".format(kraus_file[kraus_file.rfind('/') + 1:-4], str(noisy_p))
        # model_path = "./saved_models/cr_custom_{}_{}".format(kraus_file[kraus_file.rfind('/') + 1:-4], str(noisy_p))
    else:
        noise_ = noise_op_mq[noise_type].__name__
        noise_ = noise_[0: noise_.index("Channel")]
        file_name = "cr_{}_{}".format(noise_, str(noisy_p))
        # model_path = "./saved_models/cr_{}_{}".format(noise_, str(noisy_p))

model_path = "./saved_models/" + file_name
NUM_QUBITS = 9
WORKING_QUBITS = cirq.GridQubit.rect(1, NUM_QUBITS)


class LossHistory(Callback):  # 继承自Callback类

    '''
    在模型开始的时候定义四个属性，每一个属性都是字典类型，存储相对应的值和epoch
    '''

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 在每一个batch结束后记录相应的值
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    # 在每一个epoch之后记录相应的值
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        '''
        loss_type：指的是 'epoch'或者是'batch'，分别表示是一个batch之后记录还是一个epoch之后记录
        '''
        iters = range(1, len(self.losses[loss_type]) + 1)
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig("./loss_figures/" + file_name + ".png")
        # plt.show()


def generate_data_circuit(data):
    qubits = WORKING_QUBITS
    output = []
    for j in range(data.shape[0]):
        circuit = cirq.Circuit()
        for k in range(data.shape[1]):
            circuit += cirq.X(qubits[k]) ** (data[j, k] / features_MAX[k])
        output.append(circuit)
    return tfq.convert_to_tensor(output)


def generate_model_circuit(variables, p, noise_type):
    qubits = WORKING_QUBITS
    qubits_num = len(qubits)
    symbols = variables
    circuit = cirq.Circuit()
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]

    if p > 1e-5:
        if noise_type == "mixed":
            l = len(noise_list)
            for q in range(qubits_num)[::l]:
                for i in range(l):
                    if q + i < qubits_num:
                        circuit += noise_op_cirq[noise_list[i]](p)(qubits[q + i])
        elif noise_type == "custom":
            # TODO
            data = load(kraus_file)
            noisy_kraus = data['kraus']
        else:
            # noise = noise_op_cirq[noise_type]
            circuit += noise_op_cirq[noise_type](p).on_each(*qubits)

        # if mixed:
        #     circuit += cirq.bit_flip(p).on_each(*qubits[::3])
        #     circuit += cirq.depolarize(p).on_each(*qubits[1::3])
        #     circuit += cirq.phase_flip(p).on_each(*qubits[2::3])
        # else:
        #     circuit += noise_op(p).on_each(*qubits)

    circuit += [cirq.XX(q1, q2) ** next(symbols) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.XX(q1, q2) ** next(symbols) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]

    circuit += cirq.X(qubits[-1]) ** next(symbols)
    circuit += cirq.Y(qubits[-1]) ** next(symbols)
    circuit += cirq.X(qubits[-1]) ** next(symbols)

    return circuit


def print_model_circuit(variables, p, noise_type):
    qubits = [i for i in range(NUM_QUBITS)]
    num_qubits = len(qubits)
    variables = [round(i, 4) for i in variables]
    # print(variables)
    symbols = iter(variables)

    circuit = Circuit()
    for q1 in qubits:
        circuit += Power(Z, next(symbols)).on(q1)

    for q1 in qubits:
        circuit += Power(Y, next(symbols)).on(q1)

    for q1 in qubits:
        circuit += Power(Z, next(symbols)).on(q1)

    if p > 1e-5:
        if noise_type == "mixed":
            l = len(noise_list)
            for q in qubits[::l]:
                for i in range(l):
                    if q + i < num_qubits:
                        circuit += noise_op_mq[noise_list[i]](p).on(q + i)
        elif noise_type == "custom":
            # TODO
            data = load(kraus_file)
            noisy_kraus = data['kraus']
        else:
            for q in qubits:
                circuit += noise_op_mq[noise_type](p).on(q)

        # if mixed:
        #     for q in qubits[::3]:
        #         circuit += BitFlipChannel(p).on(q)
        #     for q in qubits[1::3]:
        #         circuit += DepolarizingChannel(p).on(q)
        #     for q in qubits[2::3]:
        #         circuit += PhaseFlipChannel(p).on(q)
        # else:
        #     for q in qubits:
        #         circuit += noise_op(p).on(q)

    for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]]):
        # circuit += XX(next(symbols)).on([q1, q2])
        circuit += Rxx(next(symbols)).on([q1, q2])

    for q1 in qubits:
        circuit += Power(Z, next(symbols)).on(q1)

    for q1 in qubits:
        circuit += Power(Y, next(symbols)).on(q1)

    for q1 in qubits:
        circuit += Power(Z, next(symbols)).on(q1)

    for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]]):
        # circuit += XX(next(symbols)).on([q1, q2])
        circuit += Rxx(next(symbols)).on([q1, q2])

    circuit += Power(X, next(symbols)).on(qubits[-1])
    circuit += Power(Y, next(symbols)).on(qubits[-1])
    circuit += Power(X, next(symbols)).on(qubits[-1])

    circuit += Measure('q{}'.format(qubits[-1])).on(qubits[-1])

    circuit.svg().to_file("./figures/{}.svg".format(file_name))


def circuit2M(p, variables, noise_type):
    qubits = WORKING_QUBITS
    num_qubits = len(qubits)
    variables = iter(variables)
    circuit = cirq.Circuit()
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    U1 = cirq.unitary(circuit)

    if p > 1e-5:
        noisy_kraus = []
        if noise_type == "mixed":
            l = len(noise_list)
            for q in range(num_qubits)[::l]:
                for i in range(l):
                    if q + i < num_qubits:
                        noisy_kraus += cirq.kraus(noise_op_cirq[noise_list[i]](p)(qubits[q + i]))
        elif noise_type == "custom":
            # TODO
            data = load(kraus_file)
            noisy_kraus = data['kraus']
        else:
            # noise = noise_op_cirq[noise_type]
            noisy_kraus = [cirq.kraus(noise_op_cirq[noise_type](p)(q)) for q in qubits]

        # if mixed:
        #     noisy_kraus = [cirq.channel(cirq.bit_flip(p)(q)) for q in qubits[::3]]
        #     noisy_kraus += [cirq.channel(cirq.depolarize(p)(q)) for q in qubits[1::3]]
        #     noisy_kraus += [cirq.channel(cirq.phase_flip(p)(q)) for q in qubits[2::3]]
        # else:
        #     noisy_kraus = [cirq.channel(noise_op(p)(q)) for q in qubits]

    circuit = cirq.Circuit()
    circuit += [cirq.XX(q1, q2) ** next(variables) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.XX(q1, q2) ** next(variables) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    U2 = cirq.unitary(circuit)

    M = U2.conj().T @ np.kron(np.eye(2 ** (num_qubits - 1)), np.array([[1., 0.], [0., 0.]])) @ U2

    if p > 1e-5:
        for j in range(num_qubits):
            N = 0
            for E in noisy_kraus[j]:
                F = np.kron(np.eye(2 ** j), np.kron(E, np.eye(2 ** (num_qubits - j - 1))))
                N = F.conj().T @ M @ F + N

            M = N

    M = U1.conj().T @ M @ U1
    return M


def make_quantum_model(p, noise_type):
    qubits = WORKING_QUBITS
    num = len(qubits)
    num_para = num * 8 + 3
    symbols = iter(sympy.symbols('qgenerator0:%d' % (num_para)))
    circuit_input = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    if p > 1e-5:
        quantum_layer = tfq.layers.NoisyPQC(
            generate_model_circuit(symbols, p, noise_type),
            cirq.Z(qubits[-1]),
            repetitions=100,
            sample_based=False
        )(circuit_input)
    else:
        quantum_layer = tfq.layers.PQC(
            generate_model_circuit(symbols),
            cirq.Z(qubits[-1])
        )(circuit_input)

    return tf.keras.Model(inputs=[circuit_input], outputs=[0.5 * (quantum_layer + tf.constant(1.))])


if not os.path.exists(model_path):
    print("start read csv")
    df = pd.read_csv("../../data/german_credit.csv")
    train_f = pd.read_csv("../../data/Training50.csv")
    test_f = pd.read_csv("../../data/Test50.csv")
    features = ['Account.Balance',
                'Payment.Status.of.Previous.Credit',
                'Purpose',
                'Value.Savings.Stocks',
                'Length.of.current.employment',
                'Sex...Marital.Status',
                'Guarantors',
                'Concurrent.Credits',
                'No.of.Credits.at.this.Bank']
    features_MAX = [3., 3., 4., 4., 4., 3., 2., 2., 2.]

    print("start convert_to_tensor")
    X_train = tf.convert_to_tensor(train_f[features], ).numpy()
    Y_train = tf.convert_to_tensor(train_f['Creditability'], ).numpy()
    X_test = tf.convert_to_tensor(test_f[features], ).numpy()
    Y_test = tf.convert_to_tensor(test_f['Creditability'], ).numpy()

    NUM_QUBITS = X_train.shape[1]
    WORKING_QUBITS = cirq.GridQubit.rect(1, NUM_QUBITS)

    print("start generate_data_circuit")
    X_train_input = generate_data_circuit(X_train)
    X_test_input = generate_data_circuit(X_test)

    print("start make_quantum_model")
    noisy_model = make_quantum_model(noisy_p, noise_type)
    noisy_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        # loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy']
    )

    history = LossHistory()

    print("================Training Start=================")
    train_history = noisy_model.fit(
        x=X_train_input,
        y=Y_train,
        batch_size=100,
        epochs=100,
        verbose=1,
        validation_data=(X_test_input, Y_test),
        callbacks=[history]
    )
    print("=================Training End==================")
    noisy_model.save(model_path)
    # history.loss_plot('epoch')
else:
    print("================Loading Model Start=================")
    noisy_model = tf.keras.models.load_model(model_path)
    print("=================Loading Model End==================")

print("===========Printing Model Circuit Start==========")
print_model_circuit(noisy_model.layers[1].get_weights()[0], noisy_p, noise_type)
print("===========Printing Model Circuit End============")

t_start = time.time()
print("\n===========The Lipschitz Constant Calculation Start============")
a, _ = np.linalg.eig(circuit2M(noisy_p, noisy_model.layers[1].get_weights()[0], noise_type))
k = np.real(max(a) - min(a))
if k != -1:
    print("Lipschitz K =", k)
else:
    print("Lipschitz K = -")
print(f"Elapsed time = {(time.time() - t_start):.4f}s")
print("============The Lipschitz Constant Calculation End=============")
