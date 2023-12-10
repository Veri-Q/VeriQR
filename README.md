RobustnessVerifier
===

A Qt application for robustness verification of quantum classifiers, implemented in c++.  

It provides a user-friendly GUI, visualizing both inputs and outputs and providing an intuitive way to verify the robustness of quantum machine learning algorithms with respect to a small disturbance of noises, derived from the surrounding environment. 

Clone or download the tool [RobustnessVerifier](https://github.com/sekauqnoom/RobustnessVerifier.git): 

```bash
git clone https://github.com/sekauqnoom/RobustnessVerifier.git
```

## Requirements
*You can compile RobustnessVerifier on Unix and Linux. The following installation instruction is based on Ubuntu 22.04*


### ubuntu dependency for qt6

```bash
# install cmake make gcc g++ clang llvm
sudo apt update -y
sudo apt-get install gcc g++

# install qt6-related dependencies
sudo apt install -y build-essential libgl1-mesa-dev gdb
sudo apt install -y pkg-config libssl-dev zlib1g-dev
sudo apt install -y libxkbcommon-dev
sudo apt install -y libvulkan-dev
sudo apt install -y wget vim bash curl git

# install qt6
sudo add-apt-repository universe
sudo apt install -y qt6*
sudo apt install -y libqt6*
```

### qt6 qmake setting

1. Select Qt6 system-wide

```bash
1) vim ~/qt6.conf
   # Add the following information and save: 
   qtchooser -install qt6 $(which qmake6)
2) sudo mv ~/qt6.conf /usr/share/qtchooser/qt6.conf
```

2. Set Qt6 as default option

```bash
sudo mkdir -p /usr/lib/$(uname -p)-linux-gnu/qt-default/qtchooser
sudo rm /usr/lib/$(uname -p)-linux-gnu/qt-default/qtchooser/default.conf
sudo ln -n /usr/share/qtchooser/qt6.conf /usr/lib/$(uname -p)-linux-gnu/qt-default/qtchooser/default.conf
sudo rm /usr/bin/qmake
sudo ln -s /usr/bin/qmake6 /usr/bin/qmake
```

3. Select Qt6 as default (place in ~/.bashrc for persistence):

```bash
1) vim ~/.bashrc
   # Add the following information and save: 
   export QT_SELECT=qt6
2) source ~/.bashrc
```

### dependency for [VeriQRobust](https://github.com/Veri-Q/Robustness) and [VeriQFair](https://github.com/Veri-Q/Fairness)

1. Follow the instructions of [Miniconda Installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install Miniconda.
2. Execute the following commands: 

```bash
sudo apt install -y libblas-dev liblapack-dev cmake python3-pip

cd RobustnessVerifier/py_module
conda create -n RobustV python=3.8
conda activate RobustV
chmod +x install.sh
./install.sh
```

###### Notice: In order to use the mindquantum.io.qasm module, we need to fix some bugs in source codes.

1. Replace  the code in the `extra_params` method in the `/yourPathTo/miniconda3/envs/RobustV/lib/python3.8/site-packages/mindquantum/io/qasm/openqasm.py` file 

    ```python
    out.append(float(pr))
    ```

    with the following code: 

    ```python
    if dtype == float:
        out.append(pr)
    else:
        out.append(float(pr))
    ```

2. Replace the code in `__init__` method in `class Power(NoneParamNonHermMat)` in the `/yourPathTo/miniconda3/envs/RobustV/lib/python3.8/site-packages/mindquantum/core/gates/basicgate.py` file: 

   ```python
   name = f'{gate}^{exponent}'
   ```

   with the following code: 

   ```python
   name = f'{gate}' + '^' + str(exponent)
   ```

## Compile RobustnessVerifier
```bash
cd RobustnessVerifier
mkdir build 
cd build
qmake ..
make -j8
./RobustnessVerifier
```

## Instructions for Use

Our application contains two parts:

- local-robustness verification (See Algorithm 1 in the paper [Robustness Verification of Quantum Classifiers](https://link.springer.com/chapter/10.1007/978-3-030-81685-8_7)). 
- global-robustness verification (See Algorithm 1 in the paper [Verifying Fairness in Quantum Machine Learning](https://link.springer.com/chapter/10.1007/978-3-031-13188-2_20)). 

### local-robustness verification

#### Inputs

To perform local-robustness verification experiments on *RobustnessVerifier*, the inputs required for RobustnessVerifier include a quantum classifier, a measurement operator, a training dataset, a decimal parameter, the quantum data type and the number of experiments. 

- The quantum classifier that users input should be well-trained, which consists of a quantum circuit with a measurement at the end. RobustnessVerifier accepts quantum classifiers of the following formats: 

  (i) A `NumPy data` file (in .npz format) which the quantum circuits, the measurement operator, and the training dataset are packaged together into. This kind of NumPy data files can be directly obtained by the data of the classifiers trained on the platform --- [Tensorflow Quantum](https://www.tensorflow.org/quantum/) of Google. RobustnessVerifier provides four quantum classifiers in the required format, including quantum bits classification, quantum phase recognition and cluster excitation detection from real world intractable physical problems, and the classification of MNIST. 

  (ii) A `OpenQASM 2.0` file (in .qasm format) which represents the quantum circuit corresponding to a quantum classifier. Quantum models trained with other hybrid quantum-classical machine learning frameworks such as MindSpore, Cirq and Qiskit can be translated into this intermediate representation. For example, RobustnessVerifier provides script for the translation of MindSpore models into the .qasm format. In this case, a `NumPy data` file which contains the measurement operator and the training dataset is also required. 

- The decimal parameter is the unit of the robust threshold value $\epsilon$, which together with the number of robustness verification experiments forms $\epsilon$  for each experiment. For example, for the case where the decimal precision and the number of experiments are `1e-3` and `3`, respectively, the `1e-3`, `2e-3`,`3e-3`-robustness of the quantum classifier will be checked in turn. 

- For the robustness verification of the MNIST classifier, RobustnessVerifier supports the generation of adversarial examples, which are depicted in `.png` images. 

### global-robustness verification

//////  TODO

## Conduct Experiments

//////  TODO

In the GUI, 

- you can click the "**Import file**" button to select a NumPy data file (with the .npz suffix)  that consists of a (well-trained) quantum classifier and corresponding training dataset to verify. And after setting all experiment parameters (unit of robust accuracy, number of experiments, quantum data type), click the "**run**" button to check the robustness of this classifier. 
- you can also open a saved runtime information file (with the .txt suffix) by selecting "**Open a result data file**" from the File menu so that you don't need to run the program again. 

