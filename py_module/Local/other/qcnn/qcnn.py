from typing import List
import jax.numpy as jnp
import jax, time
from jax.scipy.linalg import expm
from jax import value_and_grad, vmap
from Adam import Adam

key = jax.random.PRNGKey(int(time.time()))

Pauli_I = jnp.array([[1, 0], [0, 1]], dtype=complex)
Pauli_X = jnp.array([[0, 1], [1, 0]], dtype=complex)
Pauli_Y = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
Pauli_Z = jnp.array([[1, 0], [0, -1]], dtype=complex)


class rotation_gate(object):
    def __init__(self, qinds: List[int], Hamiltonian: jnp.DeviceArray) -> None:
        self.qinds = qinds
        self.Hamiltonian = Hamiltonian

    def __call__(self, parameter: float) -> jnp.DeviceArray:
        return expm(-1j * parameter * self.Hamiltonian)


class RX(rotation_gate):
    def __init__(self, q: int) -> None:
        super().__init__([q], Pauli_X / 2)

    def __call__(self, parameter: float):
        return super().__call__(parameter)

    def to_qasm(self, parameter: float):
        return "rx" + f"({parameter}) q[{self.qinds[0]}]"


class RY(rotation_gate):
    def __init__(self, q: int) -> None:
        super().__init__([q], Pauli_Y / 2)

    def __call__(self, parameter: float):
        return super().__call__(parameter)

    def to_qasm(self, parameter: float):
        return "ry" + f"({parameter}) q[{self.qinds[0]}]"


class RZ(rotation_gate):
    def __init__(self, q: int) -> None:
        super().__init__([q], Pauli_Z / 2)

    def __call__(self, parameter: float):
        return super().__call__(parameter)

    def to_qasm(self, parameter: float):
        return "rz" + f"({parameter}) q[{self.qinds[0]}]"


class CRZ(rotation_gate):
    def __init__(self, q1: int, q2: int) -> None:
        super().__init__([q1, q2], jnp.kron(jnp.array([[0, 0], [0, 1]]), Pauli_X) / 2)

    def __call__(self, parameter: float):
        return super().__call__(parameter)

    def to_qasm(self, parameter: float):
        return "crz" + f"({parameter}) q[{self.qinds[0]}], q[{self.qinds[1]}]"


class qcnn_block(object):
    def __init__(self, gates: List[rotation_gate]) -> None:
        self.gates = gates


class qcnn_conv(qcnn_block):
    def __init__(self, qstart: int, qend: int) -> None:
        gates: List[rotation_gate] = []
        for j in range(qstart, qend - 1, 2):
            gates += [RX(j), RZ(j), RX(j), RX(j + 1), RZ(j + 1), RX(j + 1)]
            gates += [CRZ(j, j + 1)]

        for j in range(qstart + 1, qend - 1, 2):
            gates += [RX(j), RZ(j), RX(j), RX(j + 1), RZ(j + 1), RX(j + 1)]
            gates += [CRZ(j + 1, j)]

        super().__init__(gates)


class qcnn_pool(qcnn_block):
    def __init__(self, sources: List[int], sinks: List[int]) -> None:
        gates: List[rotation_gate] = []
        for (j, k) in list(zip(sources, sinks)):
            gates += [RX(j), RZ(j), RX(j), RX(k), RZ(k), RX(k)]
            gates += [CRZ(k, j)]
            gates += [RX(j), RZ(j), RX(j), RX(k), RZ(k), RX(k)]
            gates += [CRZ(j, k)]

        super().__init__(gates)


class qcnn_single(qcnn_block):
    def __init__(self, j: int) -> None:
        gates = [RX(j), RZ(j), RX(j)]

        super().__init__(gates)


class qcnn(object):
    def __init__(self, nqubits: int) -> None:
        gates: List[rotation_gate] = []
        qstart = 0
        qend = nqubits
        while qstart < qend - 2:
            gates += qcnn_conv(qstart, qend).gates
            qmid = ((qend - qstart + 1) >> 1) + qstart
            gates += qcnn_pool(list(range(qstart, qmid)), list(range(qmid, qend))).gates
            qstart = qmid

        gates += qcnn_single(qend - 1).gates
        ngates = len(gates)

        tr_axes1: List[List[int]] = []
        tr_axes2: List[List[int]] = []
        shapes: List[List[int]] = []
        for j in range(ngates):
            gate = gates[j]
            inds = gate.qinds
            tr_axes1.append(inds + [k for k in range(nqubits) if not (k in inds)])
            axes: list[int] = [0 for k in range(nqubits)]
            for k in range(nqubits):
                axes[tr_axes1[j][k]] = k

            tr_axes2.append(axes)
            shapes.append([1 << (len(inds)), 1 << (nqubits - len(inds))])

        self.nqubits = nqubits
        self.gates = gates
        self.ngates = ngates
        self.shape = [2] * nqubits
        self.shapes = shapes
        self.tr_axes1 = tr_axes1
        self.tr_axes2 = tr_axes2
        self.parameters = jax.random.uniform(key, (ngates,)) * 4 * jnp.pi
        self.opt = Adam(sign=-1)

    def __eval__(self, parameters: jnp.DeviceArray, x: jnp.DeviceArray):
        x = x.reshape(self.shape)
        for j in range(self.ngates):
            gate = self.gates[j]
            u = gate(parameters[j])
            x = u @ jnp.transpose(x, self.tr_axes1[j]).reshape(self.shapes[j])
            x = jnp.transpose(x.reshape(self.shape), self.tr_axes2[j])

        x = x.reshape(((1 << (self.nqubits - 1)), 2)) @ jnp.array([[1], [0]])
        return jnp.real(jnp.sum(jnp.conj(x) * x))

    def predict(self, x: jnp.DeviceArray):
        return self.__eval__(self.parameters, x)

    def train(self, x_train, y_train, x_test=None, y_test=None, epochs: int = 100):
        batched_predict = vmap(self.__eval__, (None, 0))
        accuracy = lambda p, x, y: jnp.mean((batched_predict(p, x) > 0.5) == y)
        loss = lambda p, x, y: jnp.mean((batched_predict(p, x) - y) ** 2)
        loss_vg = value_and_grad(loss, (0,))
        for j in range(epochs):
            tt = time.time()
            t, dt = loss_vg(self.parameters, x_train, y_train)
            a_train = accuracy(self.parameters, x_train, y_train)
            a_test = accuracy(self.parameters, x_test, y_test)
            self.parameters += self.opt(dt[0])
            print(f"Epoch {j} in {time.time() - tt:.2f} sec")
            print("loss: ", t)
            print("train accuracy: ", a_train)
            print("test accuracy: ", a_test)

    def to_qasm(self):
        qasm = """OPENQASM 2.0;\ninclude "qelib1.inc";"""
        qasm += f"\nqreg q[{self.nqubits}];\ncreg c[1];"
        for j in range(self.ngates):
            qasm += "\n" + self.gates[j].to_qasm(self.parameters[j]) + ";"

        qasm += f"\nmeasure q[{self.nqubits - 1}] -> c[0];"

        return qasm
