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

VeriQR provides several labeled datasets, all of which are encoded into quantum data by different quantum encoders and trained on Mindspore or Tensorflow Quantum platforms to generate QML models for verification. In total, we verify the local-robustness of 41 quantum classifiers, including: 

-  The model $qubit$, used to identify the region of a quantum state in the X-Z plane of the Bloch sphere. 
- The model $iris$, trained on the [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) that is encoded into mixed states of 4 qubits via Instantaneous Quantum Polynomial (IQP) encoding, is used for classifying irises of different subgenera. 
- The models $mnist$, trained on the well-known MNIST dataset that is encoded into pure states of 8 qubits using amplitude encoding, is used for classifying handwritten digits, which can be any two of numbers ranging from 0 to 9. 
- The model $fashion$, trained on the well-known Fashion MNIST dataset that is encoded into pure states of 8 qubits via amplitude encoding, is used to classify images of two fashion products, T-shirts and ankle boots. 
-  The models $tfi$, trained on the data points of 4 qubits and of 8 qubits in the [TFI\_chain dataset](https://tensorflow.google.cn/quantum/api_docs/python/tfq/datasets/tfi_chain) respectively using the approach in [the paper](https://www.scipost.org/SciPostPhysLectNotes.61), are used for identifying the wavefunction at different phases in a quantum many-body system. 

For each of these models, we perform noisy quantum simulation by adding a noise on each qubit at the end of the circuit with a specified probability $0 \leq p \leq 1$, which can be any one of the four supported noises, namely *depolarizing*, *phase flip*, *bit flip*, and *mixed* (a mixture of the first three) introduced in ~\cite{nielsen2001quantum}. Then we set different adversarial disturbance parameters $\varepsilon$, each of which indicates the $\varepsilon-robustness$ to be verified. We demonstrate *VeriQR* with the 50 models and the full verification results are shown in the following table, where $RA$ indicates the robust accuracy of classifiers, and $VT$ is the verification time: 

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">#Qubits</th>
    <th rowspan="2">Noise Type</th>
    <th rowspan="2">p </th>
    <th rowspan="2">ε</th>
    <th colspan="2">Rough Verification</th>
    <th colspan="2">Accurate Verification</th>
  </tr>
  <tr>
    <th>RA (%)</th>
    <th>VT (sec.)</th>
    <th>RA (%)</th>
    <th>VT (sec.)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="4">qubit</td>
    <td rowspan="4">1</td>
    <td rowspan="4">depolarizing</td>
    <td rowspan="2">0.001</td>
    <td>0.001</td>
    <td>88.12</td>
    <td>0.0020</td>
    <td>90.00</td>
    <td>0.7798</td>
  </tr>
  <tr>
    <td>0.003</td>
    <td>58.75</td>
    <td>0.0020</td>
    <td>59.62</td>
    <td>2.6098</td>
  </tr>
  <tr>
    <td rowspan="2">0.05</td>
    <td>0.0005</td>
    <td>98.12</td>
    <td>0.0018</td>
    <td>99.88</td>
    <td>0.1209</td>
  </tr>
  <tr>
    <td>0.001</td>
    <td>86.12</td>
    <td>0.0019</td>
    <td>90.00</td>
    <td>0.8621</td>
  </tr>
  <tr>
    <td rowspan="4">iris</td>
    <td rowspan="4">4</td>
    <td rowspan="4">depolarizing</td>
    <td rowspan="2">0.01</td>
    <td>0.001</td>
    <td>98.75</td>
    <td>0.0010</td>
    <td>100</td>
    <td>0.1529</td>
  </tr>
  <tr>
    <td>0.01</td>
    <td>92.50</td>
    <td>0.0010</td>
    <td>100</td>
    <td>5.0708</td>
  </tr>
  <tr>
    <td rowspan="2">0.05</td>
    <td>0.005</td>
    <td>97.50</td>
    <td>0.0011</td>
    <td>100</td>
    <td>0.2546</td>
  </tr>
  <tr>
    <td>0.01</td>
    <td>91.25</td>
    <td>0.0011</td>
    <td>100</td>
    <td>3.2335</td>
  </tr>
  <tr>
    <td rowspan="4">fashion</td>
    <td rowspan="4">8</td>
    <td rowspan="4">phase flip</td>
    <td rowspan="2">0.01</td>
    <td>0.001</td>
    <td>90.60</td>
    <td>0.8519</td>
    <td>97.40</td>
    <td>8.9210</td>
  </tr>
  <tr>
    <td>0.01</td>
    <td>54.40</td>
    <td>0.7487</td>
    <td>89.10</td>
    <td>176.9451</td>
  </tr>
  <tr>
    <td rowspan="2">0.05</td>
    <td>0.001</td>
    <td>90.60</td>
    <td>0.8369</td>
    <td>97.40</td>
    <td>60.0099</td>
  </tr>
  <tr>
    <td>0.005</td>
    <td>71.90</td>
    <td>0.7713</td>
    <td>93.20</td>
    <td>101.3314</td>
  </tr>
  <tr>
    <td rowspan="4">mnist (4&amp;9)</td>
    <td rowspan="4">8</td>
    <td rowspan="4">bit flip</td>
    <td rowspan="2">0.001</td>
    <td>0.001</td>
    <td>92.20</td>
    <td>0.5289</td>
    <td>93.70</td>
    <td>10.6985</td>
  </tr>
  <tr>
    <td>0.003</td>
    <td>82.60</td>
    <td>0.6346</td>
    <td>88.00</td>
    <td>50.3715</td>
  </tr>
  <tr>
    <td rowspan="2">0.01</td>
    <td>0.0001</td>
    <td>97.10</td>
    <td>0.7101</td>
    <td>97.70</td>
    <td>2.7965</td>
  </tr>
  <tr>
    <td>0.001</td>
    <td>91.90</td>
    <td>0.5112</td>
    <td>93.70</td>
    <td>14.1884</td>
  </tr>
  <tr>
    <td rowspan="4">tfi</td>
    <td rowspan="4">4</td>
    <td rowspan="4">mixed</td>
    <td rowspan="2">0.01</td>
    <td>0.001</td>
    <td>94.38</td>
    <td>0.0025</td>
    <td>100</td>
    <td>0.7577</td>
  </tr>
  <tr>
    <td>0.005</td>
    <td>86.25</td>
    <td>0.0034</td>
    <td>100</td>
    <td>1.8267</td>
  </tr>
  <tr>
    <td rowspan="2">0.05</td>
    <td>0.005</td>
    <td>85.78</td>
    <td>0.0034</td>
    <td>100</td>
    <td>1.7682</td>
  </tr>
  <tr>
    <td>0.01</td>
    <td>79.22</td>
    <td>0.0042</td>
    <td>100</td>
    <td>2.6395</td>
  </tr>
  <tr>
    <td rowspan="4">tfi</td>
    <td rowspan="4">8</td>
    <td rowspan="4">bit flip</td>
    <td rowspan="2">0.01</td>
    <td>0.001</td>
    <td>98.44</td>
    <td>0.7339</td>
    <td>100</td>
    <td>1.3427</td>
  </tr>
  <tr>
    <td>0.01</td>
    <td>90.31</td>
    <td>0.7591</td>
    <td>100</td>
    <td>4.4057</td>
  </tr>
  <tr>
    <td rowspan="2">0.05</td>
    <td>0.005</td>
    <td>92.97</td>
    <td>0.8704</td>
    <td>100</td>
    <td>4.8931</td>
  </tr>
  <tr>
    <td>0.01</td>
    <td>89.22</td>
    <td>0.8339</td>
    <td>100</td>
    <td>6.5601</td>
  </tr>
  <tr>
    <td>mnist01</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.0005</td>
    <td>0.0001</td>
    <td>97.80</td>
    <td>0.9794</td>
    <td>98.50</td>
    <td>5.8255</td>
  </tr>
  <tr>
    <td>mnist02</td>
    <td>8</td>
    <td>depolarizing</td>
    <td>0.0005</td>
    <td>0.005</td>
    <td>62.60</td>
    <td>30.2940</td>
    <td>74.90</td>
    <td>107.9118</td>
  </tr>
  <tr>
    <td>mnist03</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.025</td>
    <td>0.003</td>
    <td>94.20</td>
    <td>0.7959</td>
    <td>97.40</td>
    <td>11.6720</td>
  </tr>
  <tr>
    <td>mnist04</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.0005</td>
    <td>0.003</td>
    <td>98.30</td>
    <td>0.8077</td>
    <td>99.50</td>
    <td>2.8234</td>
  </tr>
  <tr>
    <td>mnist05</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.025</td>
    <td>0.001</td>
    <td>99.90</td>
    <td>5.8606</td>
    <td>100</td>
    <td>5.9724</td>
  </tr>
  <tr>
    <td>mnist06</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.01</td>
    <td>0.0003</td>
    <td>100</td>
    <td>5.3062</td>
    <td>100</td>
    <td>5.3063</td>
  </tr>
  <tr>
    <td>mnist07</td>
    <td>8</td>
    <td>bit flip</td>
    <td>0.005</td>
    <td>0.0003</td>
    <td>97.40</td>
    <td>1.1405</td>
    <td>97.80</td>
    <td>6.7732</td>
  </tr>
  <tr>
    <td>mnist08</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.01</td>
    <td>0.0003</td>
    <td>98.80</td>
    <td>0.9005</td>
    <td>99.60</td>
    <td>2.7806</td>
  </tr>
  <tr>
    <td>mnist09</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.01</td>
    <td>0.0005</td>
    <td>82.70</td>
    <td>0.8899</td>
    <td>87.40</td>
    <td>39.4768</td>
  </tr>
  <tr>
    <td>mnist12</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.05</td>
    <td>0.003</td>
    <td>74.60</td>
    <td>0.8202</td>
    <td>77.90</td>
    <td>61.6236</td>
  </tr>
  <tr>
    <td>mnist13</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.075</td>
    <td>0.0003</td>
    <td>98.60</td>
    <td>0.9784</td>
    <td>98.80</td>
    <td>4.2581</td>
  </tr>
  <tr>
    <td>mnist14</td>
    <td>8</td>
    <td>depolarizing</td>
    <td>0.075</td>
    <td>0.03</td>
    <td>85.00</td>
    <td>0.7144</td>
    <td>87.20</td>
    <td>44.0592</td>
  </tr>
  <tr>
    <td>mnist15</td>
    <td>8</td>
    <td>bit flip</td>
    <td>0.0001</td>
    <td>0.03</td>
    <td>66.70</td>
    <td>0.8398</td>
    <td>66.80</td>
    <td>92.4636</td>
  </tr>
  <tr>
    <td>mnist16</td>
    <td>8</td>
    <td>bit flip</td>
    <td>0.05</td>
    <td>0.0003</td>
    <td>99.90</td>
    <td>0.8424</td>
    <td>99.90</td>
    <td>1.1047</td>
  </tr>
  <tr>
    <td>mnist17</td>
    <td>8</td>
    <td>bit flip</td>
    <td>0.05</td>
    <td>0.01</td>
    <td>80.30</td>
    <td>0.8068</td>
    <td>88.90</td>
    <td>134.6892</td>
  </tr>
  <tr>
    <td>mnist18</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.001</td>
    <td>0.03</td>
    <td>68.30</td>
    <td>0.7612</td>
    <td>68.30</td>
    <td>81.6266</td>
  </tr>
  <tr>
    <td>mnist19</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.01</td>
    <td>0.075</td>
    <td>55.60</td>
    <td>4.4499</td>
    <td>75.50</td>
    <td>81.2824</td>
  </tr>
  <tr>
    <td>mnist23</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.075</td>
    <td>0.01</td>
    <td>54.00</td>
    <td>0.7597</td>
    <td>58.10</td>
    <td>106.9021</td>
  </tr>
  <tr>
    <td>mnist24</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.005</td>
    <td>0.01</td>
    <td>43.70</td>
    <td>0.7639</td>
    <td>44.30</td>
    <td>135.0514</td>
  </tr>
  <tr>
    <td>mnist25</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.0005</td>
    <td>0.003</td>
    <td>100</td>
    <td>0.7725</td>
    <td>100</td>
    <td>0.7725</td>
  </tr>
  <tr>
    <td>mnist26</td>
    <td>8</td>
    <td>depolarizing</td>
    <td>0.075</td>
    <td>0.0003</td>
    <td>99.90</td>
    <td>0.5375</td>
    <td>99.90</td>
    <td>0.7692</td>
  </tr>
  <tr>
    <td>mnist27</td>
    <td>8</td>
    <td>bit flip</td>
    <td>0.001</td>
    <td>0.0003</td>
    <td>100</td>
    <td>1.0876</td>
    <td>100</td>
    <td>1.0877</td>
  </tr>
  <tr>
    <td>mnist28</td>
    <td>8</td>
    <td>bit flip</td>
    <td>0.001</td>
    <td>0.0003</td>
    <td>98.90</td>
    <td>0.8244</td>
    <td>99.40</td>
    <td>2.6280</td>
  </tr>
  <tr>
    <td>mnist29</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.0001</td>
    <td>0.003</td>
    <td>99.90</td>
    <td>0.7754</td>
    <td>100</td>
    <td>0.8467</td>
  </tr>
  <tr>
    <td>mnist34</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.01</td>
    <td>0.05</td>
    <td>74.40</td>
    <td>4.7773</td>
    <td>97.60</td>
    <td>28.4311</td>
  </tr>
  <tr>
    <td>mnist35</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.025</td>
    <td>0.075</td>
    <td>50.00</td>
    <td>4.3067</td>
    <td>52.50</td>
    <td>122.3814</td>
  </tr>
  <tr>
    <td>mnist36</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.005</td>
    <td>0.001</td>
    <td>100</td>
    <td>4.3917</td>
    <td>100</td>
    <td>4.3917</td>
  </tr>
  <tr>
    <td>mnist37</td>
    <td>8</td>
    <td>bit flip</td>
    <td>0.05</td>
    <td>0.001</td>
    <td>100</td>
    <td>0.7519</td>
    <td>100</td>
    <td>0.7520</td>
  </tr>
  <tr>
    <td>mnist38</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.025</td>
    <td>0.0005</td>
    <td>99.80</td>
    <td>4.5474</td>
    <td>99.80</td>
    <td>5.0365</td>
  </tr>
  <tr>
    <td>mnist39</td>
    <td>8</td>
    <td>depolarizing</td>
    <td>0.05</td>
    <td>0.001</td>
    <td>100</td>
    <td>0.5051</td>
    <td>100</td>
    <td>0.5051</td>
  </tr>
  <tr>
    <td>mnist45</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.01</td>
    <td>0.003</td>
    <td>100</td>
    <td>4.0323</td>
    <td>100</td>
    <td>4.0323</td>
  </tr>
  <tr>
    <td>mnist46</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.01</td>
    <td>0.01</td>
    <td>99.00</td>
    <td>4.0782</td>
    <td>99.80</td>
    <td>5.1361</td>
  </tr>
  <tr>
    <td>mnist47</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.0005</td>
    <td>0.003</td>
    <td>100</td>
    <td>0.7727</td>
    <td>100</td>
    <td>0.7727</td>
  </tr>
  <tr>
    <td>mnist48</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.05</td>
    <td>0.03</td>
    <td>45.60</td>
    <td>5.9031</td>
    <td>45.70</td>
    <td>137.4534</td>
  </tr>
  <tr>
    <td>mnist49</td>
    <td>8</td>
    <td>bit flip</td>
    <td>0.01</td>
    <td>0.003</td>
    <td>81.60</td>
    <td>0.7540</td>
    <td>88.00</td>
    <td>34.1436</td>
  </tr>
  <tr>
    <td>mnist56</td>
    <td>8</td>
    <td>bit flip</td>
    <td>0.025</td>
    <td>0.001</td>
    <td>100</td>
    <td>0.7607</td>
    <td>100</td>
    <td>0.7607</td>
  </tr>
  <tr>
    <td>mnist57</td>
    <td>8</td>
    <td>depolarizing</td>
    <td>0.0001</td>
    <td>0.0003</td>
    <td>100</td>
    <td>0.6417</td>
    <td>100</td>
    <td>0.6417</td>
  </tr>
  <tr>
    <td>mnist58</td>
    <td>8</td>
    <td>phase flip</td>
    <td>0.05</td>
    <td>0.01</td>
    <td>98.00</td>
    <td>1.0737</td>
    <td>99.70</td>
    <td>3.4832</td>
  </tr>
  <tr>
    <td>mnist59</td>
    <td>8</td>
    <td>bit flip</td>
    <td>0.05</td>
    <td>0.003</td>
    <td>100</td>
    <td>0.8439</td>
    <td>100</td>
    <td>0.8439</td>
  </tr>
  <tr>
    <td>mnist67</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.005</td>
    <td>0.03</td>
    <td>95.80</td>
    <td>4.4031</td>
    <td>99.60</td>
    <td>9.3411</td>
  </tr>
  <tr>
    <td>mnist68</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.0001</td>
    <td>0.075</td>
    <td>50</td>
    <td>4.2098</td>
    <td>59.90</td>
    <td>114.2690</td>
  </tr>
  <tr>
    <td>mnist69</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.05</td>
    <td>0.0005</td>
    <td>98.70</td>
    <td>4.2002</td>
    <td>99.40</td>
    <td>6.5185</td>
  </tr>
  <tr>
    <td>mnist78</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.025</td>
    <td>0.05</td>
    <td>90.20</td>
    <td>4.4387</td>
    <td>98.70</td>
    <td>15.4179</td>
  </tr>
  <tr>
    <td>mnist79</td>
    <td>8</td>
    <td>bit flip</td>
    <td>0.025</td>
    <td>0.0003</td>
    <td>100</td>
    <td>0.8480</td>
    <td>100</td>
    <td>0.8480</td>
  </tr>
  <tr>
    <td>mnist89</td>
    <td>8</td>
    <td>mixed</td>
    <td>0.005</td>
    <td>0.075</td>
    <td>71.10</td>
    <td>4.6005</td>
    <td>94.80</td>
    <td>37.9945</td>
  </tr>
</tbody>
</table>


## Global Robustness Verification

For *global-robustness*, we also add different levels of noise to each quantum model. We tested 10 quantum models on *VeriQR*, covering QCNN, Quantum Approximate Optimization Algorithms (QAOA), Variational Quantum Eigensolver (VQE) and other algorithms models, including: 

- The model $aci$, trained on the [Adult-Income dataset](https://archive.ics.uci.edu/dataset/2/adult) for income prediction. 
- The model $fct$, trained on a [dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) that contains credit transactions made by European cardholders, used to detect fraudulent credit card transactions.  
- The model $cr$, trained on the [dataset ](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) for classifying people described by a set of attributes as good or bad credit risks. 
- The models $ehc$ of 6, 8, 10, 12 qubits are obtained from the experiments in [the paper](https://www.science.org/doi/abs/10.1126/science.abb9811), used for calculating the binding energy of hydrogen chains. 
- The model named $qaoa$ is used for hardware grid problems in [the paper](https://www.nature.com/articles/s41567-020-01105-y).
- The models $iris$ and $fashion$ are same as the one in **Local-robustness Verification**. 

The full experimental results are shown in the following table: 

<table>
<thead>
  <tr>
    <th>model</th>
    <th>#qubits</th>
    <th>noise</th>
    <th>p </th>
    <th>(ε, δ)</th>
    <th>K*</th>
    <th>robust</th>
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
    <td>0.98600</td>
    <td>NO</td>
    <td>1.28</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.005</td>
    <td>(0.03, 0.0075)</td>
    <td>0.98044</td>
    <td>NO</td>
    <td>2.04</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.0001</td>
    <td>(0.005, 0.005)</td>
    <td>0.99939</td>
    <td>YES</td>
    <td>1.35</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.0001</td>
    <td>(0.03, 0.005)</td>
    <td>0.99957</td>
    <td>NO</td>
    <td>1.50</td>
  </tr>
  <tr>
    <td rowspan="4">ehc</td>
    <td rowspan="4">6</td>
    <td>bit flip</td>
    <td>0.05</td>
    <td>(0.001, 0.0005)</td>
    <td>0.89991</td>
    <td>NO</td>
    <td>5.55</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.075</td>
    <td>(0.001, 0.0001)</td>
    <td>0.89871</td>
    <td>NO</td>
    <td>4.81</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.0001</td>
    <td>(0.005, 0.003)</td>
    <td>0.99999</td>
    <td>NO</td>
    <td>3.33</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.01</td>
    <td>(0.0003, 0.0005)</td>
    <td>0.99953</td>
    <td>YES</td>
    <td>15.32</td>
  </tr>
  <tr>
    <td rowspan="4">ehc</td>
    <td rowspan="4">8</td>
    <td>bit flip</td>
    <td>0.0001</td>
    <td>(0.0003, 0.0075)</td>
    <td>0.99976</td>
    <td>YES</td>
    <td>5.71</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.05</td>
    <td>(0.001, 0.0075)</td>
    <td>0.93305</td>
    <td>YES</td>
    <td>6.71</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.025</td>
    <td>(0.075, 0.0003)</td>
    <td>0.99968</td>
    <td>NO</td>
    <td>8.39</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.0005</td>
    <td>(0.005, 0.005)</td>
    <td>0.99906</td>
    <td>YES</td>
    <td>5.40</td>
  </tr>
  <tr>
    <td rowspan="4">fashion</td>
    <td rowspan="4">8</td>
    <td>bit flip</td>
    <td>0.005</td>
    <td>(0.075, 0.005)</td>
    <td>0.98987</td>
    <td>NO</td>
    <td>2.07</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.025</td>
    <td>(0.03, 0.003)</td>
    <td>0.96274</td>
    <td>NO</td>
    <td>2.74</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.025</td>
    <td>(0.005, 0.0003)</td>
    <td>0.98313</td>
    <td>NO</td>
    <td>2.31</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.075</td>
    <td>(0.0005, 0.0075)</td>
    <td>0.88444</td>
    <td>YES</td>
    <td>4.22</td>
  </tr>
  <tr>
    <td rowspan="4">aci</td>
    <td rowspan="4">8</td>
    <td>bit flip</td>
    <td>0.0001</td>
    <td>(0.003, 0.0001)</td>
    <td>0.99985</td>
    <td>NO</td>
    <td>2.67</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.025</td>
    <td>(0.03, 0.0005)</td>
    <td>0.92419</td>
    <td>NO</td>
    <td>6.87</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.05</td>
    <td>(0.05, 0.001)</td>
    <td>0.85972</td>
    <td>NO</td>
    <td>6.39</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.005</td>
    <td>(0.005, 0.005)</td>
    <td>0.98325</td>
    <td>YES</td>
    <td>2.21</td>
  </tr>
  <tr>
    <td rowspan="4">fct</td>
    <td rowspan="4">9</td>
    <td>bit flip</td>
    <td>0.05</td>
    <td>(0.075, 0.003)</td>
    <td>0.97685</td>
    <td>NO</td>
    <td>4.31</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.05</td>
    <td>(0.0003, 0.0001)</td>
    <td>0.92476</td>
    <td>NO</td>
    <td>7.26</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.01</td>
    <td>(0.01, 0.0075)</td>
    <td>0.98274</td>
    <td>NO</td>
    <td>18.40</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.05</td>
    <td>(0.075, 0.0075)</td>
    <td>0.92949</td>
    <td>NO</td>
    <td>8.01</td>
  </tr>
  <tr>
    <td rowspan="4">cr</td>
    <td rowspan="4">9</td>
    <td>bit flip</td>
    <td>0.025</td>
    <td>(0.01, 0.0005)</td>
    <td>0.93820</td>
    <td>NO</td>
    <td>6.28</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.005</td>
    <td>(0.075, 0.005)</td>
    <td>0.98520</td>
    <td>NO</td>
    <td>2.65</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.025</td>
    <td>(0.0003, 0.0001)</td>
    <td>0.93776</td>
    <td>NO</td>
    <td>5.76</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.025</td>
    <td>(0.0001, 0.0001)</td>
    <td>0.94961</td>
    <td>YES</td>
    <td>10.13</td>
  </tr>
  <tr>
    <td rowspan="4">qaoa</td>
    <td rowspan="4">10</td>
    <td>bit flip</td>
    <td>0.005</td>
    <td>(0.05, 0.0005)</td>
    <td>0.98520</td>
    <td>NO</td>
    <td>4.81</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.0001</td>
    <td>(0.01, 0.003)</td>
    <td>0.99965</td>
    <td>NO</td>
    <td>3.95</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.005</td>
    <td>(0.075, 0.0075)</td>
    <td>0.98517</td>
    <td>NO</td>
    <td>4.85</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.001</td>
    <td>(0.03, 0.0075)</td>
    <td>0.99656</td>
    <td>NO</td>
    <td>22.21</td>
  </tr>
  <tr>
    <td rowspan="4">ehc</td>
    <td rowspan="4">10</td>
    <td>bit flip</td>
    <td>0.075</td>
    <td>(0.05, 0.0003)</td>
    <td>0.85276</td>
    <td>NO</td>
    <td>73.52</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.0005</td>
    <td>(0.03, 0.001)</td>
    <td>0.99924</td>
    <td>NO</td>
    <td>27.37</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.01</td>
    <td>(0.0003, 0.0075)</td>
    <td>0.99859</td>
    <td>YES</td>
    <td>118.26</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.0001</td>
    <td>(0.005, 0.001)</td>
    <td>0.99977</td>
    <td>NO</td>
    <td>11.29</td>
  </tr>
  <tr>
    <td rowspan="4">ehc</td>
    <td rowspan="4">12</td>
    <td>bit flip</td>
    <td>0.005</td>
    <td>(0.0005, 0.0003)</td>
    <td>0.98966</td>
    <td>NO</td>
    <td>55.67</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.0005</td>
    <td>(0.0001, 0.005)</td>
    <td>0.99926</td>
    <td>YES</td>
    <td>18.32</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.075</td>
    <td>(0.001, 0.0075)</td>
    <td>0.99885</td>
    <td>YES</td>
    <td>1581.10</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.001</td>
    <td>(0.01, 0.0001)</td>
    <td>0.99984</td>
    <td>NO</td>
    <td>102.32</td>
  </tr>
</tbody>
</table>


### Experimental Comparison

Here is an experimental comparison against a baseline implementation without tensors for global robustness verification, where TN and Baseline represent tensor-based and matrix-based implementation methods, respectively. We can see that when the number of qubits is less than 10, the tensor-based implementation does not outperforms the matrix-based implementation. However, when the number of qubits is large (e.g., in *qaoa* and *ehc* models with 12 qubits), the speed of the tensor-based implementation is significantly higher than that of the matrix-based implementation.

<table>
<thead>
  <tr>
    <th rowspan="2">model</th>
    <th rowspan="2">#qubits</th>
    <th rowspan="2">noise</th>
    <th rowspan="2">p </th>
    <th rowspan="2">(ε, δ)</th>
    <th rowspan="2">robust</th>
    <th colspan="2">TN</th>
    <th colspan="2">Baseline</th>
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
    <td>NO</td>
    <td>0.98600</td>
    <td>1.28</td>
    <td>0.99976</td>
    <td>0.02</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.005</td>
    <td>(0.03, 0.0075)</td>
    <td>NO</td>
    <td>0.98044</td>
    <td>2.04</td>
    <td>0.93864</td>
    <td>0.02</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.0001</td>
    <td>(0.005, 0.005)</td>
    <td>YES</td>
    <td>0.99939</td>
    <td>1.35</td>
    <td>0.92925</td>
    <td>0.02</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.0001</td>
    <td>(0.03, 0.005)</td>
    <td>NO</td>
    <td>0.99957</td>
    <td>1.50</td>
    <td>0.98565</td>
    <td>0.00</td>
  </tr>
  <tr>
    <td rowspan="4">ehc</td>
    <td rowspan="4">6</td>
    <td>bit flip</td>
    <td>0.05</td>
    <td>(0.001, 0.0005)</td>
    <td>NO</td>
    <td>0.89991</td>
    <td>5.55</td>
    <td>0.87652</td>
    <td>0.03</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.075</td>
    <td>(0.001, 0.0001)</td>
    <td>NO</td>
    <td>0.89871</td>
    <td>4.81</td>
    <td>0.87557</td>
    <td>1.38</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.0001</td>
    <td>(0.005, 0.003)</td>
    <td>NO</td>
    <td>0.99999</td>
    <td>3.33</td>
    <td>0.98657</td>
    <td>1.97</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.01</td>
    <td>(0.0003, 0.0005)</td>
    <td>YES</td>
    <td>0.99953</td>
    <td>15.32</td>
    <td>0.91278</td>
    <td>2.08</td>
  </tr>
  <tr>
    <td rowspan="4">ehc</td>
    <td rowspan="4">8</td>
    <td>bit flip</td>
    <td>0.0001</td>
    <td>(0.0003, 0.0075)</td>
    <td>YES</td>
    <td>0.99976</td>
    <td>5.71</td>
    <td>0.97100</td>
    <td>1.48</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.05</td>
    <td>(0.001, 0.0075)</td>
    <td>YES</td>
    <td>0.93305</td>
    <td>6.71</td>
    <td>0.99087</td>
    <td>1.88</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.025</td>
    <td>(0.075, 0.0003)</td>
    <td>NO</td>
    <td>0.99968</td>
    <td>8.39</td>
    <td>0.96781</td>
    <td>2.70</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.0005</td>
    <td>(0.005, 0.005)</td>
    <td>YES</td>
    <td>0.99906</td>
    <td>5.40</td>
    <td>0.95905</td>
    <td>0.19</td>
  </tr>
  <tr>
    <td rowspan="4">fashion</td>
    <td rowspan="4">8</td>
    <td>bit flip</td>
    <td>0.005</td>
    <td>(0.075, 0.005)</td>
    <td>NO</td>
    <td>0.98987</td>
    <td>2.07</td>
    <td>0.90009</td>
    <td>0.16</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.025</td>
    <td>(0.03, 0.003)</td>
    <td>NO</td>
    <td>0.96274</td>
    <td>2.74</td>
    <td>0.90000</td>
    <td>1.36</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.025</td>
    <td>(0.005, 0.0003)</td>
    <td>NO</td>
    <td>0.98313</td>
    <td>2.31</td>
    <td>1</td>
    <td>2.76</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.075</td>
    <td>(0.0005, 0.0075)</td>
    <td>YES</td>
    <td>0.88444</td>
    <td>4.22</td>
    <td>0.99989</td>
    <td>2.59</td>
  </tr>
  <tr>
    <td rowspan="4">aci</td>
    <td rowspan="4">8</td>
    <td>bit flip</td>
    <td>0.0001</td>
    <td>(0.003, 0.0001)</td>
    <td>NO</td>
    <td>0.99985</td>
    <td>2.67</td>
    <td>0.99980</td>
    <td>0.15</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.025</td>
    <td>(0.03, 0.0005)</td>
    <td>NO</td>
    <td>0.92419</td>
    <td>6.87</td>
    <td>0.93333</td>
    <td>0.22</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.05</td>
    <td>(0.05, 0.001)</td>
    <td>NO</td>
    <td>0.85972</td>
    <td>6.39</td>
    <td>1</td>
    <td>3.33</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.005</td>
    <td>(0.005, 0.005)</td>
    <td>YES</td>
    <td>0.98325</td>
    <td>2.21</td>
    <td>0.99938</td>
    <td>0.20</td>
  </tr>
  <tr>
    <td rowspan="4">fct</td>
    <td rowspan="4">9</td>
    <td>bit flip</td>
    <td>0.05</td>
    <td>(0.075, 0.003)</td>
    <td>NO</td>
    <td>0.97685</td>
    <td>4.31</td>
    <td>0.85238</td>
    <td>0.80</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.05</td>
    <td>(0.0003, 0.0001)</td>
    <td>NO</td>
    <td>0.92476</td>
    <td>7.26</td>
    <td>0.99933</td>
    <td>3.92</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.01</td>
    <td>(0.01, 0.0075)</td>
    <td>NO</td>
    <td>0.98274</td>
    <td>18.40</td>
    <td>1</td>
    <td>3.57</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.05</td>
    <td>(0.075, 0.0075)</td>
    <td>NO</td>
    <td>0.92949</td>
    <td>8.01</td>
    <td>0.99981</td>
    <td>0.99</td>
  </tr>
  <tr>
    <td rowspan="4">cr</td>
    <td rowspan="4">9</td>
    <td>bit flip</td>
    <td>0.025</td>
    <td>(0.01, 0.0005)</td>
    <td>NO</td>
    <td>0.93820</td>
    <td>6.28</td>
    <td>0.99001</td>
    <td>2.92</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.005</td>
    <td>(0.075, 0.005)</td>
    <td>NO</td>
    <td>0.98520</td>
    <td>2.65</td>
    <td>0.99933</td>
    <td>5.88</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.025</td>
    <td>(0.0003, 0.0001)</td>
    <td>NO</td>
    <td>0.93776</td>
    <td>5.76</td>
    <td>1</td>
    <td>4.19</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.025</td>
    <td>(0.0001, 0.0001)</td>
    <td>YES</td>
    <td>0.94961</td>
    <td>10.13</td>
    <td>0.99997</td>
    <td>6.89</td>
  </tr>
  <tr>
    <td rowspan="4">qaoa</td>
    <td rowspan="4">10</td>
    <td>bit flip</td>
    <td>0.005</td>
    <td>(0.05, 0.0005)</td>
    <td>NO</td>
    <td>0.98520</td>
    <td>4.81</td>
    <td>0.98995</td>
    <td>10.01</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.0001</td>
    <td>(0.01, 0.003)</td>
    <td>NO</td>
    <td>0.99965</td>
    <td>3.95</td>
    <td>0.99976</td>
    <td>13.88</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.005</td>
    <td>(0.075, 0.0075)</td>
    <td>NO</td>
    <td>0.98517</td>
    <td>4.85</td>
    <td>0.99454</td>
    <td>9.82</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.001</td>
    <td>(0.03, 0.0075)</td>
    <td>NO</td>
    <td>0.99656</td>
    <td>22.21</td>
    <td>0.99892</td>
    <td>16.57</td>
  </tr>
  <tr>
    <td rowspan="4">ehc</td>
    <td rowspan="4">10</td>
    <td>bit flip</td>
    <td>0.075</td>
    <td>(0.05, 0.0003)</td>
    <td>NO</td>
    <td>0.85276</td>
    <td>73.52</td>
    <td>0.98791</td>
    <td>7.01</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.0005</td>
    <td>(0.03, 0.001)</td>
    <td>NO</td>
    <td>0.99924</td>
    <td>27.37</td>
    <td>0.95652</td>
    <td>16.08</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.01</td>
    <td>(0.0003, 0.0075)</td>
    <td>YES</td>
    <td>0.99859</td>
    <td>118.26</td>
    <td>0.98452</td>
    <td>18.45</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.0001</td>
    <td>(0.005, 0.001)</td>
    <td>NO</td>
    <td>0.99977</td>
    <td>11.29</td>
    <td>0.88763</td>
    <td>8.39</td>
  </tr>
  <tr>
    <td rowspan="4">ehc</td>
    <td rowspan="4">12</td>
    <td>bit flip</td>
    <td>0.005</td>
    <td>(0.0005, 0.0003)</td>
    <td>NO</td>
    <td>0.98966</td>
    <td>55.67</td>
    <td>0.99055</td>
    <td>822.58</td>
  </tr>
  <tr>
    <td>depolarizing</td>
    <td>0.0005</td>
    <td>(0.0001, 0.005)</td>
    <td>YES</td>
    <td>0.99926</td>
    <td>18.32</td>
    <td>0.98075</td>
    <td>1779.02</td>
  </tr>
  <tr>
    <td>phase flip</td>
    <td>0.075</td>
    <td>(0.001, 0.0075)</td>
    <td>YES</td>
    <td>0.99885</td>
    <td>1581.10</td>
    <td>0.99968</td>
    <td>838.70</td>
  </tr>
  <tr>
    <td>mixed</td>
    <td>0.001</td>
    <td>(0.01, 0.0001)</td>
    <td>NO</td>
    <td>0.99984</td>
    <td>102.32</td>
    <td>0.99967</td>
    <td>1295.06</td>
  </tr>
</tbody>
</table>

