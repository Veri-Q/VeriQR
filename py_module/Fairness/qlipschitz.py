import time
import cirq
import tensornetwork as tn
import jax
import jax.numpy as jnp
from jax import jit

import numpy as np

jax.config.update('jax_platform_name', 'cpu')
tn.set_default_backend("jax")


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
        print(moment)
        for op in moment.operations:
            left_start_inds = [f"li{qubits_frontier[q]}q{q}" for q in op.qubits]
            right_start_inds = [f"ri{qubits_frontier[q]}q{q}" for q in op.qubits]
            for q in op.qubits:
                qubits_frontier[q] += 1
            left_end_inds = [f'li{qubits_frontier[q]}q{q}' for q in op.qubits]
            right_end_inds = [f'ri{qubits_frontier[q]}q{q}' for q in op.qubits]
            try:
                # unitary
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
                print(c.tensor)

            except:
                # noise
                noisy_kraus = jnp.array(cirq.channel(op))
                noisy_kraus_d = jnp.array([E.conj().T for E in cirq.channel(op)])

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

                print(d.tensor)
                print(e.tensor)

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
    v0 = None
    start0 = time.time()
    for j in range(N):
        start = time.time()
        v, e = mv(v)
        print('v1: ', v)
        print('e1: ', e)
        print('iter %d/%d, %.8f' % (j, N, e))
        # print('iter %d/%d, %.8f, elapsed time: %.4fs' % (j, N, e, time.time() - start), end='\r')
        if (time.time() - start0) / 60 / 60 > 5:
            print("\n!!Time Out!!")
            return -1
        if jnp.abs(e - e0) < 1e-6:
            break

        v0 = v
        e0 = e

    print("===============================================")
    return v, e


def smallest_eigenvalue(nqs, mv, N):
    key = jax.random.PRNGKey(int(100 * time.time()))
    print("=========Evaluate smallest eigenvalue==========")
    v = jax.random.uniform(key, [2 ** nqs])
    v = v / norm_jit(v)
    e0 = 1.
    v0 = None
    start0 = time.time()
    for j in range(N):
        start = time.time()
        v, e = mv(v)
        print('iter %d/%d, %.8f' % (j, N, 1 - e))
        # print('iter %d/%d, %.8f, elapsed time: %.4fs' % (j, N, 1 - e, time.time() - start), end='\r')
        if (time.time() - start0) / 60 / 60 > 5:
            print("\n!!Time Out!!")
            return -1
        if jnp.abs(e - e0) < 1e-6:
            # v0 = v
            # e0 = e
            break

        v0 = v
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


qubits = cirq.GridQubit.rect(1, 1)
model_circuit = cirq.Circuit(cirq.X(qubits[0]) ** 0.5, cirq.depolarize(0.01)(qubits[0]))
# model_circuit = cirq.Circuit(cirq.X(qubits[0]))
print(model_circuit)
measurement = np.array([[1., 0.], [0., 0.]])

try:
    model_circuit.unitary()
    print("no noise")
    model_circuit = cirq.Circuit()
    k, bias_kernel = lipschitz(model_circuit, qubits, measurement)

except:
    k, bias_kernel = lipschitz(model_circuit, qubits, measurement)

print('The Lipschitz constant is ', k)
print('The bias kernel is: ({}, {})'.format(bias_kernel[0], bias_kernel[1]))
