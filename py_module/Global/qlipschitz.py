import time
import cirq
import tensornetwork as tn
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import gc
import os
import sys
import csv
import signal
from contextlib import contextmanager
from random import choice, uniform
from numpy import load
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers
from cirq.contrib.qasm_import import circuit_from_qasm
from mindquantum.io.qasm.openqasm import OpenQASM
from mindquantum.core.gates import BitFlipChannel, DepolarizingChannel, PhaseFlipChannel, Measure
from mindquantum.core.circuit import Circuit

jax.config.update('jax_platform_name', 'cpu')
tn.set_default_backend("jax")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds=3600):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def qasm2mq(qasm_str):
    circuit = OpenQASM().from_string(qasm_str)
    if circuit.parameterized:
        val_list = []
        for param in circuit.params_name:
            # print(param)
            param = param.replace('pi', str(np.pi)).replace('π', str(np.pi))
            # print("param = {}, num = {}".format(param, float(param)))
            val_list.append(float(param))
        pr = dict(zip(circuit.params_name, val_list))  # 获取线路参数
        circuit = circuit.apply_value(pr)

    return circuit


def qasm2cirq_by_qiskit(file):
    with open(file, 'r') as f:
        qasm_str = f.read()

    # qasm to qiskit
    circ = QuantumCircuit.from_qasm_str(qasm_str)
    circ.remove_final_measurements()
    circ = RemoveBarriers()(circ)
    # qiskit to qasm
    qasm_str = circ.inverse().qasm()

    # qasm to cirq
    circuit = circuit_from_qasm(qasm_str)
    qubits = sorted(circuit.all_qubits())

    return qubits, circuit, qasm_str


def circuit_to_tensor(circuit, all_qubits, measurement):
    """
    convert a quantum circuit model to tensor network
    circuit: The quantum circuit written with cirq
    all_qubits: The total qubits, not only the working qubits of input circuit
    """
    qubits = sorted(circuit.all_qubits())
    qubits_frontier = {q: 0 for q in qubits}
    left_edge = {q: 0 for q in all_qubits}
    right_edge = {q: 0 for q in all_qubits}
    all_qnum = len(all_qubits)

    nodes_set = []

    # Measurement
    Measurement = [jnp.eye(2)] * (all_qnum - 1) + [measurement]
    for j in range(len(Measurement)):
        left_inds = f'li{0}q{all_qubits[j]}'
        right_inds = f'ri{0}q{all_qubits[j]}'
        a = tn.Node(Measurement[j], axis_names=[left_inds, right_inds])
        # print(a.axis_names)
        nodes_set.append(a)
        left_edge[all_qubits[j]] = a[left_inds]
        right_edge[all_qubits[j]] = a[right_inds]

    # circuit
    for moment in circuit.moments:
        # print(moment)
        for op in moment.operations:
            left_start_inds = [f"li{qubits_frontier[q]}q{q}" for q in op.qubits]
            right_start_inds = [f"ri{qubits_frontier[q]}q{q}" for q in op.qubits]
            for q in op.qubits:
                qubits_frontier[q] += 1
            left_end_inds = [f'li{qubits_frontier[q]}q{q}' for q in op.qubits]
            right_end_inds = [f'ri{qubits_frontier[q]}q{q}' for q in op.qubits]
            try:
                # unitary
                # print("no noise")
                op.gate._has_unitary_()
                U = jnp.array(cirq.unitary(op).reshape((2,) * 2 * len(op.qubits)))
                U_d = jnp.array(cirq.unitary(op).conj().T.reshape((2,) * 2 * len(op.qubits)))

                b = tn.Node(U_d, axis_names=left_end_inds + left_start_inds)
                nodes_set.append(b)
                # U^\dagger M
                for j in range(len(op.qubits)):
                    b[left_start_inds[j]] ^ left_edge[op.qubits[j]]
                    left_edge[op.qubits[j]] = b[left_end_inds[j]]

                c = tn.Node(U, axis_names=right_start_inds + right_end_inds)

                nodes_set.append(c)
                # U^\dagger rho U
                for j in range(len(op.qubits)):
                    c[right_start_inds[j]] ^ right_edge[op.qubits[j]]
                    right_edge[op.qubits[j]] = c[right_end_inds[j]]
                # print(c.tensor)

            except:
                # noise
                # print("noisy")
                noisy_kraus = jnp.array(cirq.kraus(op))
                noisy_kraus_d = jnp.array([E.conj().T for E in cirq.kraus(op)])

                kraus_inds = [f'ki{qubits_frontier[q]}q{q}' for q in op.qubits]

                d = tn.Node(noisy_kraus_d, axis_names=kraus_inds + left_end_inds + left_start_inds)
                nodes_set.append(d)
                e = tn.Node(noisy_kraus, axis_names=kraus_inds + right_start_inds + right_end_inds)
                nodes_set.append(e)

                # E^\dagger E
                for j in range(len(kraus_inds)):
                    d[kraus_inds[j]] ^ e[kraus_inds[j]]

                # E rho E^\dagger
                for j in range(len(op.qubits)):
                    e[right_start_inds[j]] ^ right_edge[op.qubits[j]]
                    right_edge[op.qubits[j]] = e[right_end_inds[j]]

                    d[left_start_inds[j]] ^ left_edge[op.qubits[j]]
                    left_edge[op.qubits[j]] = d[left_end_inds[j]]

                # print(d.tensor)
                # print(e.tensor)

    return nodes_set, [left_edge[q] for q in all_qubits], [right_edge[q] for q in all_qubits]


def model_to_mv(model_circuit, qubits, measurement):
    measurement = jnp.array(measurement)

    def mv1(v):
        nodes_set, left_edge, right_edge = circuit_to_tensor(model_circuit, qubits, measurement)
        node_v = tn.Node(v.reshape([2] * len(qubits)), axis_names=[edge.name for edge in left_edge])
        nodes_set.append(node_v)
        # A(|psi>)
        for j in range(len(qubits)):
            right_edge[j] ^ node_v[left_edge[j].name]

        y = tn.contractors.auto(nodes_set, left_edge).tensor.reshape([2 ** len(qubits)])
        e = jnp.linalg.norm(y)
        return y / e, e

    def mv2(v):
        nodes_set, left_edge, right_edge = circuit_to_tensor(model_circuit, qubits, jnp.eye(2) - measurement)
        node_v = tn.Node(v.reshape([2] * len(qubits)), axis_names=[edge.name for edge in left_edge])
        nodes_set.append(node_v)
        for j in range(len(qubits)):
            right_edge[j] ^ node_v[left_edge[j].name]

        y = tn.contractors.auto(nodes_set, left_edge).tensor.reshape([2 ** len(qubits)])
        e = jnp.linalg.norm(y)
        return y / e, e

    return len(qubits), jit(mv1), jit(mv2)


norm_jit = jit(jnp.linalg.norm)


def largest_eigenvalue(nqs, mv, N):
    key = jax.random.PRNGKey(int(100 * time.time()))
    print("==========Evaluate largest eigenvalue==========")
    v = jax.random.uniform(key, [2 ** nqs])
    v = v / norm_jit(v)
    e0 = 1.
    start0 = time.time()
    for j in range(N):
        start = time.time()
        v, e = mv(v)
        print('iter %d/%d, %.8f, elapsed time: %.4fs' % (j, N, e, time.time() - start))
        if (time.time() - start0) / 60 / 60 > 5:
            print("\n!!Time Out!!")
            return -1
        if jnp.abs(e - e0) < 1e-6:
            break

        e0 = e

    print("===============================================")
    return v, e


def smallest_eigenvalue(nqs, mv, N):
    key = jax.random.PRNGKey(int(100 * time.time()))
    print("=========Evaluate smallest eigenvalue==========")
    v = jax.random.uniform(key, [2 ** nqs])
    v = v / norm_jit(v)
    e0 = 1.
    start0 = time.time()
    for j in range(N):
        start = time.time()
        v, e = mv(v)
        print('iter %d/%d, %.8f, elapsed time: %.4fs' % (j, N, 1 - e, time.time() - start))
        if (time.time() - start0) / 60 / 60 > 5:
            print("\n!!Time Out!!")
            return -1
        if jnp.abs(e - e0) < 1e-6:
            # v0 = v
            # e0 = e
            break

        e0 = e

    print("===============================================")
    return v, 1 - e


def lipschitz(model_circuit, qubits, measurement):
    n, mv1, mv2 = model_to_mv(model_circuit, qubits, measurement)
    v1, e1 = largest_eigenvalue(n, mv1, 200)
    if e1 == -1:
        return -1
    v2, e2 = smallest_eigenvalue(n, mv2, 200)
    if e2 == -1:
        return -1

    phi = v1
    psi = v2
    k = e1 - e2
    return k, (phi, psi)


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


def noisy_circuit_from_qasm(file, noise_type, noise_list, kraus_file, p):
    qubits, circuit_cirq, qasm_str = qasm2cirq_by_qiskit(file)

    circuit_mq = qasm2mq(qasm_str)

    all_measures = []
    for gate in circuit_mq:
        # print(type(gate))
        if type(gate) is Measure:
            all_measures.append(gate)

    if circuit_mq.has_measure_gate:
        circuit_mq = circuit_mq.remove_measure()

    # noise_op_cirq_ = noise_op_cirq[noise_type]
    # noise_op_mq_ = noise_op_mq[noise_type]
    if p > 1e-7:
        if noise_type == "mixed":
            l = len(noise_list)
            for q in range(cirq.num_qubits())[::l]:
                for i in range(l):
                    circuit_cirq += noise_op_cirq[noise_list[i]](p).on(qubits[q + i])
                    circuit_mq += noise_op_mq[noise_list[i]](p).on(q + i)
        elif noise_type == "custom":
            # TODO
            data = load(kraus_file)
            noisy_kraus = data['kraus']
        else:
            # noise = noise_op_cirq[noise_type]
            circuit_cirq += noise_op_cirq[noise_type](p).on_each(*qubits)
            for q in range(circuit_mq.n_qubits):
                circuit_mq += noise_op_mq[noise_type](p).on(q)

        # if noise_type == "mixed":
        #     circuit_cirq += cirq.bit_flip(p).on_each(*qubits[::3])
        #     circuit_cirq += cirq.depolarize(p).on_each(*qubits[1::3])
        #     circuit_cirq += cirq.phase_flip(p).on_each(*qubits[2::3])
        #     n_qubits = range(circuit_mq.n_qubits)
        #     for q in n_qubits[::3]:
        #         circuit_mq += BitFlipChannel(p).on(q)
        #     for q in n_qubits[1::3]:
        #         circuit_mq += DepolarizingChannel(p).on(q)
        #     for q in n_qubits[2::3]:
        #         circuit_mq += PhaseFlipChannel(p).on(q)
        # else:
        #     circuit_cirq += noise_op_(p).on_each(*qubits)
        #     for q in range(circuit_mq.n_qubits):
        #         circuit_mq += noise_op_mq_(p).on(q)

    for m in all_measures:
        circuit_mq += m
    return qubits, circuit_cirq, circuit_mq


def calculate_lipschitz(file, noise_type, noise_list, kraus_file, p, file_name):
    qubits, model_circuit, circuit_mq = noisy_circuit_from_qasm(file, noise_type, noise_list, kraus_file, p)

    # file_name = "{}_{}_{}".format(file[file.rfind("/") + 1: file.index(".qasm")], noise_type, str(p))
    # print("file_name: {}".format(file_name))
    circuit_mq.svg().to_file("./model_circuits/{}.svg".format(file_name))
    print("===========Printing Model Circuit End============\n")

    measurement = np.array([[1., 0.], [0., 0.]])

    print("===========The Lipschitz Constant Calculation Start============")
    start_time = time.time()
    k, bias_kernel = lipschitz(model_circuit, qubits, measurement)
    total_time = time.time() - start_time

    print('Circuit: %s' % file)
    print('Noise configuration: {}, {}'.format(noise_type, p))
    print('Lipschitz K =', k)
    print('Elapsed time = %.4fs' % total_time)
    # print('The bias kernel is: (\n{},\n {})'.format(bias_kernel[0], bias_kernel[1]))
    print("============The Lipschitz Constant Calculation End=============")
    return k, total_time, bias_kernel


# def verification(file, noise_type, p, epsilon, delta):
#     k, total_time, bias_kernel = calculate_lipschitz(file, noise_type, p)
#     start = time.time()
#     if delta >= k * epsilon:
#         return True, total_time + time.time() - start, []
#     return False, total_time + time.time() - start, bias_kernel


def verification(k, epsilon, delta):
    if delta >= k * epsilon:
        print('This model is ({}, {})-robust.'.format(epsilon, delta))
        # print('YES')
        return True

    print('This model is not ({}, {})-robust.'.format(epsilon, delta))
    # print('NO')
    return False


noise_ops = ["phase_flip", "depolarizing", "bit_flip"]

if str(sys.argv[1]) != "verify":
    # python qlipschitz.py ./qasm_models/HFVQE/hf_6_0_5.qasm phase_flip 0.0001
    qasm_file = str(sys.argv[1])
    model_name = qasm_file[qasm_file.rfind("/")+1:-5]
    arg_num = len(sys.argv)
    noise_list = []
    kraus_file = ''
    if arg_num <= 2:  # random noise
        noise_type = choice(noise_ops)
        noisy_p = float(round(uniform(0, 0.2), 5))  # 随机数的精度round(数值，精度)
        file_name = "{}_{}_{}".format(model_name, noise_type, str(noisy_p))
    else:
        noise_type = str(sys.argv[2])
        noisy_p = float(sys.argv[arg_num-1])
        if noise_type == 'mixed':
            noise_list = [i for i in sys.argv[3: arg_num - 2]]
            noise_list_ = [noise_op_mq[i].__name__ for i in noise_list]
            noise_list_ = [i[0: i.index("Channel")] for i in noise_list_]
            print("noise_list: ", noise_list)
            file_name = "{}_mixed_{}_{}".format(model_name, '_'.join(noise_list_), str(noisy_p))
        elif noise_type == 'custom':
            kraus_file = sys.argv[3]
            file_name = "{}_custom_{}_{}".format(model_name, kraus_file[kraus_file.rfind('/') + 1:-4], str(noisy_p))
        else:
            noise_ = noise_op_mq[noise_type].__name__
            noise_ = noise_[0: noise_.index("Channel")]
            file_name = "{}_{}_{}".format(model_name, noise_, str(noisy_p))
    k, total_time, bias_kernel = calculate_lipschitz(qasm_file, noise_type, noise_list, kraus_file, noisy_p, file_name)
else:
    # python qlipschitz.py verify k epsilon delta
    k = float(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    # flag, k, bias_kernel, total_time = verification(epsilon, delta)
    print("===========The Global Verification Start============")
    flag = verification(k, epsilon, delta)
    print("===========The Global Verification End============")


# qubits = cirq.GridQubit.rect(1, 1)
# model_circuit = cirq.Circuit(cirq.X(qubits[0]) ** 0.5, cirq.depolarize(0.01)(qubits[0]))
# # model_circuit = cirq.Circuit(cirq.X(qubits[0]))
# print(model_circuit)
