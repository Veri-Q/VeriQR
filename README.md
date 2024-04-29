VeriQR
===

A Qt application for robustness verification of quantum classifiers, implemented in c++.  

It provides a user-friendly GUI, visualizing both inputs and outputs and providing an intuitive way to verify the robustness of quantum machine learning algorithms with respect to a small disturbance of noises, derived from the surrounding environment. 

Clone or download the tool [VeriQR](https://github.com/sekauqnoom/RobustnessVerifier.git): 

```bash
git clone https://github.com/Veri-Q/VeriQR.git
```

## Requirements
*You can compile VeriQR on Unix and Linux. The following installation instruction is based on Ubuntu 22.04*


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

### dependency for python module

1. Follow the instructions of [Miniconda Installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install Miniconda.
2. Execute the following commands: 

```bash
sudo apt install -y libblas-dev liblapack-dev cmake python3-pip

cd VeriQR/py_module
conda create -n VeriQR python=3.9.0
conda activate VeriQR
chmod +x install.sh
./install.sh
```

###### Notice: In order to use the mindquantum.io.qasm module, we need to fix some bugs in source codes.

1. Replace  the code in the `extra_params` method in the `/yourPathTo/miniconda3/envs/VeriQR/lib/python3.9/site-packages/mindquantum/io/qasm/openqasm.py` file 

    ```python
    def extra_params(cmd, dtype=float):
        """Get gate parameters."""
        matches = re.findall(r'\((.*)\)', cmd)
        out = []
        for i in matches:
            for j in i.split(','):
                pr = j.strip()
                if dtype == str:
                    out.append(pr)
                else:
                    if '*' in pr:
                        pr = pr.replace('pi', str(np.pi)).replace('π', str(np.pi))
                        pr = [float(i.strip()) for i in pr.split('*')]
                        out.append(pr[0] * pr[1])
                    elif '/' in pr:
                        pr = pr.replace('pi', str(np.pi)).replace('π', str(np.pi))
                        pr = [float(i.strip()) for i in pr.split('/')]
                        out.append(pr[0] / pr[1])
                    else:
                        out.append(float(pr))
        return out
    ```

    with the following code: 

    ```python
    def extra_params(cmd, dtype=float):
        """Get gate parameters."""
        matches = re.findall(r'\((.*)\)', cmd)
        out = []
        for i in matches:
            for j in i.split(','):
                pr = j.strip()
                if dtype == str:
                    out.append(pr)
                else:
                    flag_1 = '*' in pr
                    flag_2 = '/' in pr
                    if flag_1:
                        pr = pr.replace('pi', str(np.pi)).replace('π', str(np.pi))
                        pr = [i.strip() for i in pr.split('*')]
                    if flag_2:
                        result = 1
                        if not flag_1:
                            pr = pr.replace('pi', str(np.pi)).replace('π', str(np.pi))
                            pr = [k.strip() for k in pr.split('/')]
                            r = float(pr[0])
                            for k in range(len(pr) - 1):
                                r = r / float(pr[k + 1])
                            result *= r
                        else:
                            for item in pr:
                                items = [k.strip() for k in item.split('/')]
                                r = float(items[0])
                                for k in range(len(items) - 1):
                                    r = r / float(items[k + 1])
                                result *= r
    
                        out.append(result)
                    else:
                        if dtype == float:
                            out.append(pr)
                        else:
                            out.append(float(pr))
        return out
    ```

2. Replace the code in `__init__` method in `class Power(NoneParamNonHermMat)` in the `/yourPathTo/miniconda3/envs/VeriQR/lib/python3.9/site-packages/mindquantum/core/gates/basicgate.py` file: 

   ```python
   name = f'{gate}^{exponent}'
   ```

   with the following code: 

   ```python
   name = f'{gate}' + '^' + str(exponent)
   ```

## Compile VeriQR

```bash
cd VeriQR
mkdir build 
cd build
# If the 'VeriQR' env has been deactivated, you need to activate it again now: 
conda activate VeriQR
qmake ..
make -j8
./VeriQR
```

## User Manual

See [veriqr_manual.pdf](https://github.com/Veri-Q/VeriQR/blob/main/veriqr_manual.pdf)

# Experimental Results

## Local Raobustness Verification

VeriQR provides several labeled datasets, all of which are encoded into quantum data by different quantum encoders and trained on Mindspore or Tensorflow Quantum platforms to generate QML models for verification. In total, we verify the local-robustness of 50 quantum classifiers, including: 

-  The model $qubit$, used to identify the region of a quantum state in the X-Z plane of the Bloch sphere. 
- The model $iris$, trained on the [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) that is encoded into mixed states of 4 qubits via Instantaneous Quantum Polynomial (IQP) encoding, is used for classifying irises of different subgenera. 
- The models $mnist$, trained on the well-known MNIST dataset that is encoded into pure states of 8 qubits using amplitude encoding, is used for classifying handwritten digits, which can be any two of numbers ranging from 0 to 9. 
- The model $fashion$, trained on the well-known Fashion MNIST dataset that is encoded into pure states of 8 qubits via amplitude encoding, is used to classify images of two fashion products, T-shirts and ankle boots. 
-  The models $tfi$, trained on the data points of 4 qubits and of 8 qubits in the [TFI\_chain dataset](https://tensorflow.google.cn/quantum/api_docs/python/tfq/datasets/tfi_chain) respectively using the approach in [the paper](https://www.scipost.org/SciPostPhysLectNotes.61), are used for identifying the wavefunction at different phases in a quantum many-body system. 

We conducted numerous experiments on different circuits for each model as outlined in the [local_results](https://github.com/Veri-Q/VeriQR/blob/main/py_module/Local/results/local_results.csv) table: 

- The noiseless ideal QML model with quantum circuit $c_0$; 
- Circuit $c_1$ created by introducing random noise at various random points in circuit $c_0$ to simulate noise effects on NISQ devices; 
- Circuit $c_2$ modified by adding specific noise with a noise level $0 \leq p \leq 1$ of four types: *depolarizing*, *phase flip*, *bit flip*, and *mixed* (a combination of the three) noise (referred to as "noisename\_p" below $c_2$), applied to each qubit after the random noise manipulation on circuit $c_1$. 

Where $RA$ indicates the robust accuracy of classifiers, and $VT$ is the verification time in seconds. 

## Global Robustness Verification

For *global-robustness*, we also add different levels of noise to each quantum model. We tested 12 QML models on *VeriQR*, covering QCNN, Quantum Approximate Optimization Algorithms (QAOA), Variational Quantum Eigensolver (VQE) and other algorithms, including: 

- The model $aci$, trained on the [Adult-Income dataset](https://archive.ics.uci.edu/dataset/2/adult) for income prediction. 
- The model $fct$, trained on a [dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) that contains credit transactions made by European cardholders, used to detect fraudulent credit card transactions.  
- The model $cr$, trained on the [dataset ](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) for classifying people described by a set of attributes as good or bad credit risks. 
- The models $ehc$ of 6, 8, 10, 12 qubits are obtained from the experiments in [the paper](https://www.science.org/doi/abs/10.1126/science.abb9811), used for calculating the binding energy of hydrogen chains. 
- The model named $qaoa$ is used for hardware grid problems in [the paper](https://www.nature.com/articles/s41567-020-01105-y).
- The models $iris$ and $fashion$ are same as the one in **Local-robustness Verification**. 

The full experimental results are shown in the [global_results](https://github.com/Veri-Q/VeriQR/blob/main/py_module/Global/results/global_results.csv) table: 

### Experimental Comparison

Here is an experimental comparison against a baseline implementation without tensors for global robustness verification, where "TN" and "Baseline" represent tensor-based and matrix-based implementation methods, respectively. 

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">#Qubits</th>
    <th rowspan="2">Noise</th>
    <th rowspan="2">p </th>
    <th rowspan="2">(ε, δ)</th>
    <th colspan="2">Baseline</th>
    <th colspan="2">TN</th>
    <th rowspan="2">robust</th>
  </tr>
  <tr>
    <th>K*</th>
    <th>time (sec.)</th>
    <th>K*</th>
    <th>time (sec.)</th>
  </tr>
</thead>
<tbody>
    <tr>
        <td rowspan="4">iris</td>
        <td rowspan="4">4</td>
        <td>bit flip</td>
        <td>0.005</td>
        <td>(0.003, 0.0001)</td>
        <td>0.98859</td>
        <td>0.01</td>
        <td>0.98733</td>
        <td>2.86</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>depolarizing</td>
        <td>0.005</td>
        <td>(0.03, 0.0075)</td>
        <td>0.98310</td>
        <td>0.01</td>
        <td>0.98077</td>
        <td>2.48</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>phase flip</td>
        <td>0.0001</td>
        <td>(0.005, 0.005)</td>
        <td>0.99943</td>
        <td>0</td>
        <td>0.99938</td>
        <td>2.39</td>
        <td>YES</td>
    </tr>
    <tr>
        <td>mixed</td>
        <td>0.0001</td>
        <td>(0.03, 0.005)</td>
        <td>0.99968</td>
        <td>0</td>
        <td>0.99961</td>
        <td>2.36</td>
        <td>NO</td>
    </tr>
    <tr>
        <td rowspan="4">ehc</td>
        <td rowspan="4">6</td>
        <td>bit flip</td>
        <td>0.05</td>
        <td>(0.001, 0.0005)</td>
        <td>0.90028</td>
        <td>0.02</td>
        <td>0.89994</td>
        <td>13.28</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>depolarizing</td>
        <td>0.075</td>
        <td>(0.001, 0.0001)</td>
        <td>0.90000</td>
        <td>0.02</td>
        <td>0.89892</td>
        <td>13.18</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>phase flip</td>
        <td>0.0001</td>
        <td>(0.005, 0.003)</td>
        <td>1</td>
        <td>0.02</td>
        <td>0.99999</td>
        <td>13.14</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>mixed</td>
        <td>0.01</td>
        <td>(0.0003, 0.0005)</td>
        <td>0.99993</td>
        <td>0.02</td>
        <td>0.99954</td>
        <td>13.58</td>
        <td>YES</td>
    </tr>
    <tr>
        <td rowspan="4">ehc</td>
        <td rowspan="4">8</td>
        <td>bit flip</td>
        <td>0.0001</td>
        <td>(0.0003, 0.0075)</td>
        <td>0.99980</td>
        <td>0.26</td>
        <td>0.99976</td>
        <td>26.17</td>
        <td>YES</td>
    </tr>
    <tr>
        <td>depolarizing</td>
        <td>0.05</td>
        <td>(0.001, 0.0075)</td>
        <td>0.93333</td>
        <td>0.26</td>
        <td>0.93304</td>
        <td>27.87</td>
        <td>YES</td>
    </tr>
    <tr>
        <td>phase flip</td>
        <td>0.025</td>
        <td>(0.075, 0.0003)</td>
        <td>1</td>
        <td>0.26</td>
        <td>0.99968</td>
        <td>28.46</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>mixed</td>
        <td>0.0005</td>
        <td>(0.005, 0.005)</td>
        <td>0.99938</td>
        <td>0.24</td>
        <td>0.99905</td>
        <td>25.75</td>
        <td>YES</td>
    </tr>
    <tr>
        <td rowspan="4">fashion</td>
        <td rowspan="4">8</td>
        <td>bit flip</td>
        <td>0.005</td>
        <td>(0.075, 0.005)</td>
        <td>0.99000</td>
        <td>0.12</td>
        <td>0.98987</td>
        <td>6.01</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>depolarizing</td>
        <td>0.025</td>
        <td>(0.03, 0.003)</td>
        <td>0.96274</td>
        <td>0.16</td>
        <td>0.96274</td>
        <td>6.37</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>phase flip</td>
        <td>0.025</td>
        <td>(0.005, 0.0003)</td>
        <td>0.98452</td>
        <td>0.13</td>
        <td>0.98313</td>
        <td>6.03</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>mixed</td>
        <td>0.075</td>
        <td>(0.0005, 0.0075)</td>
        <td>0.88444</td>
        <td>0.14</td>
        <td>0.88444</td>
        <td>5.95</td>
        <td>YES</td>
    </tr>
    <tr>
        <td rowspan="4">aci</td>
        <td rowspan="4">8</td>
        <td>bit flip</td>
        <td>0.0001</td>
        <td>(0.003, 0.0001)</td>
        <td>0.99985</td>
        <td>0.18</td>
        <td>0.99985</td>
        <td>6.44</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>depolarizing</td>
        <td>0.025</td>
        <td>(0.03, 0.0005)</td>
        <td>0.92640</td>
        <td>0.25</td>
        <td>0.92440</td>
        <td>7.70</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>phase flip</td>
        <td>0.05</td>
        <td>(0.05, 0.001)</td>
        <td>0.88450</td>
        <td>0.19</td>
        <td>0.85990</td>
        <td>8.58</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>mixed</td>
        <td>0.005</td>
        <td>(0.005, 0.005)</td>
        <td>0.98384</td>
        <td>0.22</td>
        <td>0.98326</td>
        <td>6.06</td>
        <td>YES</td>
    </tr>
    <tr>
        <td rowspan="4">fct</td>
        <td rowspan="4">9</td>
        <td>bit flip</td>
        <td>0.05</td>
        <td>(0.075, 0.003)</td>
        <td>0.99024</td>
        <td>0.98</td>
        <td>0.97683</td>
        <td>13.89</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>depolarizing</td>
        <td>0.05</td>
        <td>(0.0003, 0.0001)</td>
        <td>0.92638</td>
        <td>0.76</td>
        <td>0.92486</td>
        <td>40.73</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>phase flip</td>
        <td>0.01</td>
        <td>(0.01, 0.0075)</td>
        <td>0.98730</td>
        <td>0.87</td>
        <td>0.98290</td>
        <td>10.45</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>mixed</td>
        <td>0.05</td>
        <td>(0.075, 0.0075)</td>
        <td>0.94531</td>
        <td>0.89</td>
        <td>0.92949</td>
        <td>9.06</td>
        <td>NO</td>
    </tr>
    <tr>
        <td rowspan="4">cr</td>
        <td rowspan="4">9</td>
        <td>bit flip</td>
        <td>0.025</td>
        <td>(0.01, 0.0005)</td>
        <td>0.93964</td>
        <td>0.65</td>
        <td>0.93819</td>
        <td>14.44</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>depolarizing</td>
        <td>0.005</td>
        <td>(0.075, 0.005)</td>
        <td>0.98637</td>
        <td>1.21</td>
        <td>0.98515</td>
        <td>6.49</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>phase flip</td>
        <td>0.025</td>
        <td>(0.0003, 0.0001)</td>
        <td>0.94753</td>
        <td>0.97</td>
        <td>0.93772</td>
        <td>9.63</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>mixed</td>
        <td>0.025</td>
        <td>(0.0001, 0.0001)</td>
        <td>0.95579</td>
        <td>0.93</td>
        <td>0.94980</td>
        <td>12.15</td>
        <td>YES</td>
    </tr>
    <tr>
        <td rowspan="4">qaoa</td>
        <td rowspan="4">10</td>
        <td>bit flip</td>
        <td>0.005</td>
        <td>(0.05, 0.0005)</td>
        <td>0.99843</td>
        <td>5.23</td>
        <td>0.98507</td>
        <td>16.98</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>depolarizing</td>
        <td>0.0001</td>
        <td>(0.01, 0.003)</td>
        <td>0.99983</td>
        <td>6.15</td>
        <td>0.99965</td>
        <td>16.10</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>phase flip</td>
        <td>0.005</td>
        <td>(0.075, 0.0075)</td>
        <td>0.99224</td>
        <td>5.14</td>
        <td>0.98516</td>
        <td>17.95</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>mixed</td>
        <td>0.001</td>
        <td>(0.03, 0.0075)</td>
        <td>0.99923</td>
        <td>4.98</td>
        <td>0.99657</td>
        <td>16.16</td>
        <td>NO</td>
    </tr>
    <tr>
        <td rowspan="4">ehc</td>
        <td rowspan="4">10</td>
        <td>bit flip</td>
        <td>0.075</td>
        <td>(0.05, 0.0003)</td>
        <td>0.85409</td>
        <td>3.37</td>
        <td>0.85262</td>
        <td>82.25</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>depolarizing</td>
        <td>0.0005</td>
        <td>(0.03, 0.001)</td>
        <td>0.99933</td>
        <td>5.69</td>
        <td>0.99924</td>
        <td>40.33</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>phase flip</td>
        <td>0.01</td>
        <td>(0.0003, 0.0075)</td>
        <td>1</td>
        <td>4.36</td>
        <td>0.99857</td>
        <td>66.67</td>
        <td>YES</td>
    </tr>
    <tr>
        <td>mixed</td>
        <td>0.0001</td>
        <td>(0.005, 0.001)</td>
        <td>0.99981</td>
        <td>5.26</td>
        <td>0.99977</td>
        <td>38.13</td>
        <td>NO</td>
    </tr>
    <tr>
        <td rowspan="4">ehc</td>
        <td rowspan="4">12</td>
        <td>bit flip</td>
        <td>0.005</td>
        <td>(0.0005, 0.0003)</td>
        <td>0.99001</td>
        <td>169.42</td>
        <td>0.98965</td>
        <td>76.77</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>depolarizing</td>
        <td>0.0005</td>
        <td>(0.0001, 0.005)</td>
        <td>0.99933</td>
        <td>253.11</td>
        <td>0.99926</td>
        <td>189.35</td>
        <td>YES</td>
    </tr>
    <tr>
        <td>phase flip</td>
        <td>0.075</td>
        <td>(0.001, 0.0075)</td>
        <td>1</td>
        <td>163.61</td>
        <td>0.99880</td>
        <td>675.50</td>
        <td>YES</td>
    </tr>
    <tr>
        <td>mixed</td>
        <td>0.001</td>
        <td>(0.01, 0.0001)</td>
        <td>0.99997</td>
        <td>195.48</td>
        <td>0.99984</td>
        <td>64.50</td>
        <td>NO</td>
    </tr>
    <tr>
        <td rowspan="4">inst</td>
        <td rowspan="4">16</td>
        <td>bit flip</td>
        <td>0.005</td>
        <td>(0.0005, 0.0003)</td>
        <td>-</td>
        <td>TO</td>
        <td>0.98009</td>
        <td>1052.73</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>depolarizing</td>
        <td>0.0005</td>
        <td>(0.0003, 0.005)</td>
        <td>-</td>
        <td>TO</td>
        <td>0.99833</td>
        <td>33.99</td>
        <td>YES</td>
    </tr>
    <tr>
        <td>phase flip</td>
        <td>0.05</td>
        <td>(0.001, 0.0075)</td>
        <td>-</td>
        <td>TO</td>
        <td>0.95131</td>
        <td>381.15</td>
        <td>YES</td>
    </tr>
    <tr>
        <td>mixed</td>
        <td>0.001</td>
        <td>(0.005, 0.0003)</td>
        <td>-</td>
        <td>TO</td>
        <td>0.99899</td>
        <td>123.25</td>
        <td>NO</td>
    </tr>
    <tr>
        <td rowspan="4">qaoa</td>
        <td rowspan="4">20</td>
        <td>bit flip</td>
        <td>0.05</td>
        <td>(0.005, 0.001)</td>
        <td>-</td>
        <td>TO</td>
        <td>0.91194</td>
        <td>2402.32</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>depolarizing</td>
        <td>0.075</td>
        <td>(0.005, 0.003)</td>
        <td>-</td>
        <td>TO</td>
        <td>0.83488</td>
        <td>433.05</td>
        <td>NO</td>
    </tr>
    <tr>
        <td>phase flip</td>
        <td>0.0005</td>
        <td>(0.0001, 0.0001)</td>
        <td>-</td>
        <td>TO</td>
        <td>0.99868</td>
        <td>70.00</td>
        <td>YES</td>
    </tr>
    <tr>
        <td>mixed</td>
        <td>0.05</td>
        <td>(0.075, 0.0003)</td>
        <td>-</td>
        <td>TO</td>
        <td>0.89682</td>
        <td>4635.55</td>
        <td>NO</td>
    </tr>
</tbody>
</table>

