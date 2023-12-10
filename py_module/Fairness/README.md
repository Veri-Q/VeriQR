# VeriQFair: Verifying Fairness in Quantum Machine Learning #

This repository contains two parts:
- An implementation for computing the Lipschitz constant of a quantum decision model (See Algorithm 1 in the paper).
- Experiment codes and data for CAV2022 Artifact Evaluation (See Section 6 in the paper).
  
[Guan, J., Fang, W. and Ying, M., 2022, August. Verifying Fairness in Quantum Machine Learning. In International Conference on Computer Aided Verification (pp. 408-429). Cham: Springer International Publishing.](https://link.springer.com/chapter/10.1007/978-3-031-13188-2_20)
## Requirements ##

- [Python3.8](https://www.python.org/).
- Python libraries: 
    * [Cirq](https://quantumai.google/cirq) for representing (noisy) quantum circuits.
    * [Tensornetwork](https://github.com/google/tensornetwork) for manipulating tensor networks.
    * [Numpy](https://numpy.org/) for linear algebra computations.
    * [Jax](https://github.com/google/jax) for just-in-time (JIT) compilation in Python.
    * [Tensorflow Quantum](https://www.tensorflow.org/quantum) for training quantum decision models.
    * [dice-ml](https://github.com/interpretml/DiCE) for DiCE adult income dataset.

## Installation (for Linux) ##

We recommend the users to use [Conda](https://docs.conda.io/en/latest/) to configure the Python environment.

### Install with Conda (Miniconda) ###
1. Follow the instructions of [Miniconda Installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install Miniconda.
2. Clone this repository and cd to it.
    ```bash
    git clone https://github.com/Veri-Q/Fairness.git && cd Fairness
    ```
3. Use Conda to create a new Conda environment:
    ```bash
    conda create -n qfairness python=3.8.12
    ```
4. Activate the above environment and use pip to install required libraries in `requirements.txt`.
    ```bash
    conda activate qfairness
    pip install -r requirements.txt
    ```

### Install with Docker ###

This repository provides a Dockerfile. The user can directly built a docker image from this Dockerfile and use it.

1. Clone this repository and cd to it.
    ```bash
    git clone https://github.com/Veri-Q/Fairness.git && cd Fairness
    ```
2. Build docker image.
    ```bash
    docker build -t qfairness .
    ```
3. Run the built image.
    ```bash
    docker run -t -i qfairness /bin/bash
    ```

🟥 For CAV2022 Artifact, we also provide a **Docker image** `qfairness.tar` on [Zenodo](https://doi.org/10.5281/zenodo.6612720). The user can load this image and run it directly:
```
docker load -i qfairness.tar && docker run -i -t qfairness /bin/bash
``` 

## Computing the Lipschitz Constant ##

The file `qlipschitz.py` in this repository is the implementation of Algorithm 1 in the paper. It provides a function `lipschitz` that accepts a quantum decision model and outputs the model's Lipschitz constant as defined in the paper. The usage of `lipschitz` in Python is as follows:
```python
from qlipschitz import lipschitz

# ...


k = lipschitz(model_circuit, qubits, measurement)
# model_circuit: the (noisy) quantum circuit descried by Cirq; It expresses the super-operator $\mathcal{E}$ in the quantum decision model.
# qubits: all (cirq) qubits used in the model; usually, qubits = model_circuit.all_qubits()
# measurement: a single qubit measurement (2x2 Hermitian matrix) on the last one of all qubits in the model; It expresses the measurement $M$ at the end of the model.

# ...
```

For example,

```python
import cirq
import numpy as np

from qlipschitz import lipschitz

qubits = cirq.GridQubit.rect(1, 1)
model_circuit = cirq.Circuit(cirq.X(qubits[0])**0.5, cirq.depolarize(0.01)(qubits[0]))
measurement = np.array([[1., 0.], [0., 0.]])

k = lipschitz(model_circuit, qubits, measurement)

print('The Lipschitz constant is ', k)
```

## Experiments (Artifact Evaluations) ##

🟥 Notice: Due to the inherent randomness in the training of quantum models, the results of repeated experiments may be numerically inconsistent. 

### A Practical Application in Finance (GC & DiCE) ###

We provide two scripts `evaluate_finance_model_gc.py` and `evaluate_finance_model_dice.py` to reproduce Table 1 in the paper. These two scripts will train a quantum decision model based on the given arguments (`<noise_type>` and  `<noisy_probability>`) and compute the Lipschtiz constant of the trained model:

1. For **German Credit** in Table 1

    ```bash
    python evaluate_finance_model_gc.py <noise_type> <noisy_probability>
    ```
2. For **Adult Income (DiCE)** in Table 1

    ```bash
    python evaluate_finance_model_dice.py <noise_type> <noisy_probability>
    ```
where `<noisy_probability>` is the probability of noise that can be valued at `0.0`, `0.01`, `0.001` and `0.0001`; `<noise_type>` is the type of noise that has four options: `phase_flip` for phase flip noise, `depolarize` for depolarize noise, `bit_flip` for bit flip noise and `mixed` for mixed noise, which is the mixture of the three aforementioned noises.

For example, running `python evaluate_finance_model_gc.py depolarize 0.0001` can reproduce the results of **German Credit** and **Depolarize** noise with probability **10^(-4)** in Table 1.

---
*🟥 Since TensorFlow Quantum is inefficient in training noisy models, we provide trained parameters for **German Credit**. The users can load the parameters and reproduce the part of **German Credit** in Table 1 by the script `evaluate_trained_model_gc.py`.*

For example, run command `python evaluate_finance_model_gc.py depolarize 0.0001` can reproduce the Lipschitz constant and evaluate the time of **German Credit** and **Depolarize** noise with probability **10^(-4)** in Table 1.

###  Scalability in the NISQ era (QCNN Models) ###

We provide a script `evaluate_qcnn_model.py` to reproduce Table 2 in the paper.

```bash
python evaluate_qcnn_model.py <qubits_num> <noise_type>
```
where `<qubits_num>` is the number of qubits (integer), and again, `<noise_type>` is the type of noise that has four options: `phase_flip` for phase flip noise, `depolarize` for depolarize noise, `bit_flip` for bit flip noise and `mixed` for the mixed one.

For example, running command `python evaluate_qcnn_model.py 25 depolarize` can reproduce the results of **25 Qubits** and **Depolarize** noise in Table 2.

---
*🟥 The server used in our experiments has 2048GB of memory. For the users who do not have a server with the same memory, you can test on a smaller number (10-15) of qubits*.
