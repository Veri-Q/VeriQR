"""Functions to convert between Cirq's internal circuit representation and
Qiskit's circuit representation.
"""

import re
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import qiskit
# from mitiq.utils import _simplify_circuit_exponents
import numpy as np
from mindquantum.io import OpenQASM
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers

# python3 qasm2npz.py qe.qasm
# qasm_file = str(argv[1])
qasm_file = "../model_and_data/test_qasm/pe_10.qasm"

QASMType = str


def cirq2qasm(circuit: cirq.Circuit) -> QASMType:
    """Returns a QASM string representing the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a QASM string.

    Returns:
        QASMType: QASM string equivalent to the input Mitiq circuit.
    """
    # Simplify exponents of gates. For example, H**-1 is simplified to H.
    #    _simplify_circuit_exponents(circuit)
    return circuit.to_qasm()


def to_qiskit(circuit: cirq.Circuit) -> qiskit.QuantumCircuit:
    """Returns a Qiskit circuit equivalent to the input Mitiq circuit. Note
    that the output circuit registers may not match the input circuit
    registers.

    Args:
        circuit: Mitiq circuit to convert to a Qiskit circuit.

    Returns:
        Qiskit.QuantumCircuit object equivalent to the input Mitiq circuit.
    """
    return qiskit.QuantumCircuit.from_qasm_str(cirq2qasm(circuit))


def _remove_qasm_barriers(qasm: QASMType) -> QASMType:
    """Returns a copy of the input QASM with all barriers removed.

    Args:
        qasm: QASM to remove barriers from.

    Note:
        According to the OpenQASM 2.X language specification
        (https://arxiv.org/pdf/1707.03429v2.pdf), "Statements are separated by
        semicolons. Whitespace is ignored. The language is case sensitive.
        Comments begin with a pair of forward slashes and end with a new line."
    """
    quoted_re = r"(?:\"[^\"]*?\")"
    statement_re = r"((?:[^;{}\"]*?" + quoted_re + r"?)*[;{}])?"
    comment_re = r"(\n?//[^\n]*(?:\n|$))?"
    statements_comments = re.findall(statement_re + comment_re, qasm)
    lines = []
    for statement, comment in statements_comments:
        if re.match(r"^\s*barrier(?:(?:\s+)|(?:;))", statement) is None:
            lines.append(statement + comment)
    return "".join(lines)


def qasm2cirq_(qasm: QASMType) -> cirq.Circuit:
    """Returns a Mitiq circuit equivalent to the input QASM string.

    Args:
        qasm: QASM string to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input QASM string.
    """
    qasm = _remove_qasm_barriers(qasm)
    return circuit_from_qasm(qasm)


def qasm2cirq(file):
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

    return qubits, circuit


def noisy_circuit_from_qasm(file, noise_op, p=0.01):
    qubits, circuit = qasm2cirq(file)

    if p > 1e-7:
        circuit += noise_op(p).on_each(*qubits)
    return qubits, circuit


def mq2qasm(circ):
    circuit_str = OpenQASM().to_string(circ)
    print(circuit_str)
    f = open("test.qasm")
    f.write(circuit_str)
    f.close()


def qasm2mq(qasm_file):
    f = open(qasm_file)
    qasm = f.read()
    f.close()
    circuit = OpenQASM().from_string(qasm)
    # print(circuit)
    # print(circuit.parameterized)
    # print(circuit.params_name)
    if circuit.parameterized:
        val_list = []
        for param in circuit.params_name:
            val_list.append(float(param))
        pr = dict(zip(circuit.params_name, val_list))  # 获取线路参数
        circuit = circuit.apply_value(pr)
        # print(pr)

    # 先保存图片, 再移除测量门
    circuit.svg().to_file(
        "./figures/" + qasm_file[qasm_file.rfind('/') + 1:-5] + "_model.svg")  # qasm_file chop '.qasm'
    if circuit.has_measure_gate:
        circuit = circuit.remove_measure()

    mat = circuit.matrix()
    # print(circuit)
    # print(mat.shape)
    kraus = np.array([mat])  # .reshape((-1, 2, mat.shape[0]))
    return kraus


def savenpz():
    a = np.arange(3)
    b = np.arange(4)
    c = np.arange(5)
    d = np.arange(6)
    np.savez('array_save.npz', kraus=a, O=b, data=c, label=d)


def cu1(p_lambda):
    return cirq.MatrixGate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j * p_lambda)]]))


def circ():
    import cirq
    from functools import reduce

    q = [cirq.NamedQubit('q' + str(i)) for i in range(11)]

    circuit = cirq.Circuit(
        cirq.H(q[0]),
        cirq.H(q[1]),
        cirq.H(q[2]),
        cirq.H(q[3]),
        cirq.H(q[4]),
        cirq.H(q[5]),
        cirq.H(q[6]),
        cirq.H(q[7]),
        cirq.H(q[8]),
        cirq.H(q[9]),
        cu1(np.pi / 512)(q[9], q[10]),
        cu1(np.pi / 256)(q[8], q[10]),
        cu1(np.pi / 128)(q[7], q[10]),
        cu1(np.pi / 64)(q[6], q[10]),
        cu1(np.pi / 32)(q[5], q[10]),
        cu1(np.pi / 16)(q[4], q[10]),
        cu1(np.pi / 8)(q[3], q[10]),
        cu1(np.pi / 4)(q[2], q[10]),
        cu1(np.pi / 2)(q[1], q[10]),
        cu1(np.pi)(q[0], q[10]),
        cirq.H(q[0]),
        cu1(-np.pi / 2)(q[0], q[1]),
        cu1(-np.pi / 4)(q[0], q[2]),
        cu1(-np.pi / 8)(q[0], q[3]),
        cu1(-np.pi / 16)(q[0], q[4]),
        cu1(-np.pi / 32)(q[0], q[5]),
        cu1(-np.pi / 64)(q[0], q[6]),
        cu1(-np.pi / 128)(q[0], q[7]),
        cu1(-np.pi / 256)(q[0], q[8]),
        cu1(-np.pi / 512)(q[0], q[9]),
        cirq.H(q[1]),
        cu1(-np.pi / 2)(q[1], q[2]),
        cu1(-np.pi / 4)(q[1], q[3]),
        cu1(-np.pi / 8)(q[1], q[4]),
        cu1(-np.pi / 16)(q[1], q[5]),
        cu1(-np.pi / 32)(q[1], q[6]),
        cu1(-np.pi / 64)(q[1], q[7]),
        cu1(-np.pi / 128)(q[1], q[8]),
        cu1(-np.pi / 256)(q[1], q[9]),
        cirq.H(q[2]),
        cu1(-np.pi / 2)(q[2], q[3]),
        cu1(-np.pi / 4)(q[2], q[4]),
        cu1(-np.pi / 8)(q[2], q[5]),
        cu1(-np.pi / 16)(q[2], q[6]),
        cu1(-np.pi / 32)(q[2], q[7]),
        cu1(-np.pi / 64)(q[2], q[8]),
        cu1(-np.pi / 128)(q[2], q[9]),
        cirq.H(q[3]),
        cu1(-np.pi / 2)(q[3], q[4]),
        cu1(-np.pi / 4)(q[3], q[5]),
        cu1(-np.pi / 8)(q[3], q[6]),
        cu1(-np.pi / 16)(q[3], q[7]),
        cu1(-np.pi / 32)(q[3], q[8]),
        cu1(-np.pi / 64)(q[3], q[9]),
        cirq.H(q[4]),
        cu1(-np.pi / 2)(q[4], q[5]),
        cu1(-np.pi / 4)(q[4], q[6]),
        cu1(-np.pi / 8)(q[4], q[7]),
        cu1(-np.pi / 16)(q[4], q[8]),
        cu1(-np.pi / 32)(q[4], q[9]),
        cirq.H(q[5]),
        cu1(-np.pi / 2)(q[5], q[6]),
        cu1(-np.pi / 4)(q[5], q[7]),
        cu1(-np.pi / 8)(q[5], q[8]),
        cu1(-np.pi / 16)(q[5], q[9]),
        cirq.H(q[6]),
        cu1(-np.pi / 2)(q[6], q[7]),
        cu1(-np.pi / 4)(q[6], q[8]),
        cu1(-np.pi / 8)(q[6], q[9]),
        cirq.H(q[7]),
        cu1(-np.pi / 2)(q[7], q[8]),
        cu1(-np.pi / 4)(q[7], q[9]),
        cirq.H(q[8]),
        cu1(-np.pi / 2)(q[8], q[9]),
        cirq.H(q[9])
    )
    print(circuit)

    # simulator = cirq.Simulator()
    # result = simulator.run(circuit, repetitions=1024)
    # result_dict = dict(result.multi_measurement_histogram(keys=[]))
    # keys = list(map(lambda arr: reduce(lambda x, y: str(x) + str(y), arr[::-1]), result_dict.keys()))
    # counts = dict(zip(keys, [value for value in result_dict.values()]))
    # print(counts)


if __name__ == "__main__":
    # qasm2cirq(qasm_file)
    qasm2mq(qasm_file)
