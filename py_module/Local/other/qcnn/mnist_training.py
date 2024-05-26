from qcnn import qcnn
import numpy as np
import jax.numpy as jnp
from PIL import Image
from sklearn.datasets import fetch_openml
from jax.numpy.linalg import norm
from mindquantum.io import OpenQASM
from mindquantum.core.gates import I

normalize = lambda v: v / norm(v)

m = np.array([[1. + 0.j, 0. + 0.j],
              [0. + 0.j, 0. + 0.j]])


def mat_m(qubits_num):
    M_0 = np.kron(np.eye(2 ** (qubits_num - 1)), m)
    return M_0


M = mat_m(8)


def qasm2mq(qasm_str):
    circuit = OpenQASM().from_string(qasm_str)
    if circuit.parameterized:
        val_list = []
        for param in circuit.params_name:
            param = param.replace('pi', str(np.pi)).replace('π', str(np.pi))
            val_list.append(float(param))
        pr = dict(zip(circuit.params_name, val_list))  # 获取线路参数
        circuit = circuit.apply_value(pr)
    return circuit


def convert_to_qcnn_data(data):
    n = data.shape[0]
    return jnp.array(
        [normalize(jnp.array(Image.fromarray(data[j].reshape(28, 28)).resize((16, 16))).reshape((256))) for j in
         range(n)])


digits = fetch_openml('mnist_784')

# for d0 in range(1, 10):
for d1 in range(1, 10):
    d0, d1 = str(0), str(d1)
    # if d0+d1 in ['12', '17', '38', '39', '49', '56', '68']:
    #     continue

    ind0, ind1 = digits.target == d0, digits.target == d1
    x0, x1 = digits.data[ind0].to_numpy(), digits.data[ind1].to_numpy()
    y0, y1 = (digits.target[ind0] == d0).to_numpy(), (digits.target[ind1] == d0).to_numpy()

    n_train = 500
    n_all = 700

    ind0, ind1 = np.random.permutation(x0.shape[0])[:n_all], np.random.permutation(x1.shape[0])[:n_all]

    x_train = convert_to_qcnn_data(np.vstack((x0[ind0[:n_train]], x1[ind1[:n_train]])))
    y_train = jnp.array(np.hstack((y0[ind0[:n_train]], y1[ind1[:n_train]])))
    print(x_train.shape)
    print(type(x_train))
    print(x_train[0])
    print(y_train.shape)
    print(type(y_train))
    print(y_train[0])
    print(type(y_train[0]))
    # print(y_train)
    x_test = convert_to_qcnn_data(np.vstack((x0[ind0[n_train:n_all]], x1[ind1[n_train:n_all]])))
    y_test = jnp.array(np.hstack((y0[ind0[n_train:n_all]], y1[ind1[n_train:n_all]])))
    print(x_test.shape)
    print(type(x_test))
    print(y_test.shape)
    print(type(y_test))

    model = qcnn(8)

    model.train(x_train, y_train, x_test, y_test)

    qasm_str = model.to_qasm()
    f = open('../../model_and_data/mnist{}.qasm'.format(d0 + d1), 'w')
    f.write(qasm_str)
    f.close()

    ansatz = qasm2mq(qasm_str)
    ansatz.svg().to_file("../../figures/mnist{}_model.svg".format(d0 + d1))

    print(M.shape)
    x_train = x_train.T
    y_train = [1 if i == True else 0 for i in y_train]
    label = np.array(y_train)
    print(x_train.shape)
    # print(x_train)
    print(label.shape)
    # print(label)
    np.savez('../../model_and_data/mnist{}_data.npz'.format(d0 + d1), O=M, data=x_train, label=label)
