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


# digits = fetch_openml('mnist_784')
def retaining(data_file):
    DATA = np.load('../../model_and_data/newdata_for_AT/{}'.format(data_file))
    x_train = jnp.array(DATA['data'].T)
    y_train = jnp.array(DATA['label'])
    print(x_train.shape)
    print(type(x_train))
    print(x_train[0])
    print(y_train.shape)
    print(type(y_train))

    model = qcnn(8)

    model.train(x_train, y_train)

    def get_trained_ansatz():
        file_name = data_file[:-4]
        qasm_str = model.to_qasm()
        f = open('../../model_and_data/{}.qasm'.format(file_name), 'w')
        f.write(qasm_str)
        f.close()

        ansatz = qasm2mq(qasm_str)
        ansatz.svg().to_file("../../figures/{}.svg".format(file_name))

        U = ansatz.matrix()
        np.savez('../../model_and_data/newmodel_by_AT/{}.npz'.format(file_name), kraus=np.array([U]))
    get_trained_ansatz()


retaining('mnist13_c0_by_0.003.npz')