import sympy
import cirq
import numpy as np
import time
import sys
import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.callbacks import Callback

from dice_ml.utils import helpers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X, Z, Y, Rxx, Power, BitFlipChannel, DepolarizingChannel, PhaseFlipChannel


noise_type = str(sys.argv[1])
noisy_p = float(sys.argv[2])
choice = str(sys.argv[3])

noise_op = {
    "phase_flip": cirq.phase_flip,
    "depolarize": cirq.depolarize,
    "bit_flip": cirq.bit_flip,
    "mixed": cirq.depolarize
}

noise_op_mq = {
    "phase_flip": PhaseFlipChannel,
    "depolarize": DepolarizingChannel,
    "bit_flip": BitFlipChannel,
    "mixed": DepolarizingChannel
}

if noise_type == "mixed":
    mixed = True
else:
    mixed = False

file_name = "dice_" + noise_type + "_" + str(sys.argv[2])
model_path = "./saved_model/" + file_name

NUM_QUBITS = 8
WORKING_QUBITS = cirq.GridQubit.rect(1, NUM_QUBITS)

features = ['workclass', 'education', 'marital_status', 'occupation', 'race', 'gender']
features_MAX = [90., 3., 7., 4., 5., 1., 1., 99.]


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
        plt.savefig("./result_figures/" + file_name + ".png")
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


def generate_model_circuit(variables, p=0., noise_op=cirq.depolarize, mixed=False):
    qubits = WORKING_QUBITS
    symbols = variables
    circuit = cirq.Circuit()
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]

    if p > 1e-5:
        if mixed:
            circuit += cirq.bit_flip(p).on_each(*qubits[::3])
            circuit += cirq.depolarize(p).on_each(*qubits[1::3])
            circuit += cirq.phase_flip(p).on_each(*qubits[2::3])
        else:
            circuit += noise_op(p).on_each(*qubits)

    circuit += [cirq.XX(q1, q2) ** next(symbols) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(symbols) for q1 in qubits]
    circuit += [cirq.XX(q1, q2) ** next(symbols) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]

    circuit += cirq.X(qubits[-1]) ** next(symbols)
    circuit += cirq.Y(qubits[-1]) ** next(symbols)
    circuit += cirq.X(qubits[-1]) ** next(symbols)

    return circuit


def print_model_circuit(variables, p=0., noise_op=DepolarizingChannel, mixed=False):
    qubits = [i for i in range(NUM_QUBITS)]
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
        if mixed:
            for q in qubits[::3]:
                circuit += BitFlipChannel(p).on(q)

            for q in qubits[1::3]:
                circuit += DepolarizingChannel(p).on(q)

            for q in qubits[2::3]:
                circuit += PhaseFlipChannel(p).on(q)
        else:
            for q in qubits:
                circuit += noise_op(p).on(q)

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

    circuit.svg().to_file("./model_circuits/circuit_{}.svg".format(file_name))


def circuit2M(p, variables, noise_op=cirq.depolarize, mixed=False):
    qubits = WORKING_QUBITS
    num = len(qubits)
    variables = iter(variables)
    circuit = cirq.Circuit()
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    U1 = cirq.unitary(circuit)

    if p > 1e-5:
        if mixed:
            noisy_kraus = [cirq.channel(cirq.bit_flip(p)(q)) for q in qubits[::3]]
            noisy_kraus += [cirq.channel(cirq.depolarize(p)(q)) for q in qubits[1::3]]
            noisy_kraus += [cirq.channel(cirq.phase_flip(p)(q)) for q in qubits[2::3]]
        else:
            noisy_kraus = [cirq.channel(noise_op(p)(q)) for q in qubits]

    circuit = cirq.Circuit()
    circuit += [cirq.XX(q1, q2) ** next(variables) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Y(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.Z(q1) ** next(variables) for q1 in qubits]
    circuit += [cirq.XX(q1, q2) ** next(variables) for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])]
    U2 = cirq.unitary(circuit)

    M = U2.conj().T @ np.kron(np.eye(2 ** (num - 1)), np.array([[1., 0.], [0., 0.]])) @ U2

    if p > 1e-5:
        for j in range(num):
            N = 0
            for E in noisy_kraus[j]:
                F = np.kron(np.eye(2 ** j), np.kron(E, np.eye(2 ** (num - j - 1))))
                N = F.conj().T @ M @ F + N

            M = N

    M = U1.conj().T @ M @ U1
    return M


def make_quantum_model(p, noise_op, mixed):
    qubits = WORKING_QUBITS
    num = len(qubits)
    num_para = num * 8 + 3
    symbols = iter(sympy.symbols('qgenerator0:%d' % (num_para)))
    circuit_input = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    if p > 1e-5:
        quantum_layer = tfq.layers.NoisyPQC(
            generate_model_circuit(symbols, p, noise_op, mixed),
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


if choice == "train":
    dataset = helpers.load_adult_income_dataset()
    target = dataset["income"]
    train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                    target,
                                                                    test_size=0.2,
                                                                    random_state=0,
                                                                    stratify=target)
    x_train = train_dataset.drop('income', axis=1)
    x_test = test_dataset.drop('income', axis=1)

    for key in features:
        x_train[key] = LabelEncoder().fit_transform(x_train[key])
        x_test[key] = LabelEncoder().fit_transform(x_test[key])

    X_train = tf.convert_to_tensor(x_train).numpy()
    Y_train = tf.convert_to_tensor(y_train).numpy()
    X_test = tf.convert_to_tensor(x_test).numpy()
    Y_test = tf.convert_to_tensor(y_test).numpy()

    idxs = tf.range(tf.shape(X_train)[0])
    ridxs = tf.random.shuffle(idxs)[:1000]
    X_train = X_train[ridxs, :]
    Y_train = Y_train[ridxs]

    idxs = tf.range(tf.shape(X_test)[0])
    ridxs = tf.random.shuffle(idxs)[:400]
    X_test = X_test[ridxs, :]
    Y_test = Y_test[ridxs]

    NUM_QUBITS = X_train.shape[1]  # 8
    WORKING_QUBITS = cirq.GridQubit.rect(1, NUM_QUBITS)

    X_train_input = generate_data_circuit(X_train)
    X_test_input = generate_data_circuit(X_test)

    noisy_model = make_quantum_model(noisy_p, noise_op[noise_type], mixed)
    noisy_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        # loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy']
    )

    history = LossHistory()

    print("================Training Start=================")
    train_history = noisy_model.fit(
        x=X_train_input,
        y=Y_train,
        batch_size=50,
        epochs=100,
        verbose=1,
        validation_data=(X_test_input, Y_test),
        callbacks=[history]
    )
    print("=================Training End==================")

    noisy_model.save(model_path)

    history.loss_plot('epoch')

elif choice == "notrain":
    print("================Loading Model Start=================")
    #    print(model_path)
    noisy_model = tf.keras.models.load_model(model_path)
    print("=================Loading Model End==================")


print("===========Printing Model Circuit Start==========")
print_model_circuit(noisy_model.layers[1].get_weights()[0], noisy_p, noise_op_mq[noise_type], mixed)
print("===========Printing Model Circuit End============")

tstart = time.time()
print("\n===========Lipschitz Constant Start============")
a, _ = np.linalg.eig(circuit2M(noisy_p, noisy_model.layers[1].get_weights()[0], noise_op[noise_type], mixed))
k = np.real(max(a) - min(a))
if k != -1:
    print("Lipschitz K = ", k)
else:
    print("Lipschitz K = -")
print(f"Elapsed time = {(time.time() - tstart):.4f}s")
print("============Lipschitz Constant End=============")
