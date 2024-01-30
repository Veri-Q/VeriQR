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

### dependency for [VeriQRobust](https://github.com/Veri-Q/Robustness) and [VeriQFair](https://github.com/Veri-Q/Fairness)

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

# Results

## Local Raobustness Verification

VeriQR provides several labeled datasets, all of which are encoded into quantum data by different quantum encoders and trained on Mindspore or Tensorflow Quantum platforms to generate QML models for verification. In total, we verify the local-robustness of 41 quantum classifiers, including: 

-  The model $qubit$, used to identify the region of a quantum state in the X-Z plane of the Bloch sphere. 
- The model $iris$, trained on the [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) that is encoded into mixed states of 4 qubits via Instantaneous Quantum Polynomial (IQP) encoding, is used for classifying irises of different subgenera. 
- The models $mnist$, trained on the well-known MNIST dataset that is encoded into pure states of 8 qubits using amplitude encoding, is used for classifying handwritten digits, which can be any two of numbers ranging from 0 to 9. 
- The model $fashion$, trained on the well-known Fashion MNIST dataset that is encoded into pure states of 8 qubits via amplitude encoding, is used to classify images of two fashion products, T-shirts and ankle boots. 
-  The models $tfi$, trained on the data points of 4 qubits and of 8 qubits in the [TFI\_chain dataset](https://tensorflow.google.cn/quantum/api_docs/python/tfq/datasets/tfi_chain) respectively using the approach in [the paper](https://www.scipost.org/SciPostPhysLectNotes.61), are used for identifying the wavefunction at different phases in a quantum many-body system. 

For each of these models, we perform noisy quantum simulation by adding a noise on each qubit at the end of the circuit with a specified probability $0 \leq p \leq 1$, which can be any one of the four supported noises, namely *depolarizing*, *phase flip*, *bit flip*, and *mixed* (a mixture of the first three) introduced in ~\cite{nielsen2001quantum}. Then we set different adversarial disturbance parameters $\varepsilon$, each of which indicates the $\varepsilon-robustness$ to be verified. We demonstrate *VeriQR* with the **50** models and the full verification results are shown in the following table, where $RA$ indicates the robust accuracy of classifiers, and $VT$ is the verification time: 

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-pb0m{border-color:inherit;text-align:center;vertical-align:bottom}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-nrix{text-align:center;vertical-align:middle}
.tg .tg-8d8j{text-align:center;vertical-align:bottom}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt" rowspan="2">Model</th>
    <th class="tg-7btt" rowspan="2">#Qubits</th>
    <th class="tg-7btt" rowspan="2">Noise Type</th>
    <th class="tg-7btt" rowspan="2">p </th>
    <th class="tg-7btt" rowspan="2">ε</th>
    <th class="tg-7btt" colspan="2">Rough Verification</th>
    <th class="tg-7btt" colspan="2">Accurate Verification</th>
  </tr>
  <tr>
    <th class="tg-7btt">RA (%)</th>
    <th class="tg-7btt">VT (sec.)</th>
    <th class="tg-7btt">RA (%)</th>
    <th class="tg-7btt">VT (sec.)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="4">qubit</td>
    <td class="tg-9wq8" rowspan="4">1</td>
    <td class="tg-9wq8" rowspan="4">depolarizing</td>
    <td class="tg-9wq8" rowspan="2">0.001</td>
    <td class="tg-pb0m">0.001</td>
    <td class="tg-pb0m">88.12</td>
    <td class="tg-pb0m">0.002</td>
    <td class="tg-pb0m">90</td>
    <td class="tg-pb0m">0.7798</td>
  </tr>
  <tr>
    <td class="tg-pb0m">0.003</td>
    <td class="tg-pb0m">58.75</td>
    <td class="tg-pb0m">0.002</td>
    <td class="tg-pb0m">59.62</td>
    <td class="tg-pb0m">2.6098</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">0.05</td>
    <td class="tg-pb0m">0.0005</td>
    <td class="tg-pb0m">98.12</td>
    <td class="tg-pb0m">0.0018</td>
    <td class="tg-pb0m">99.88</td>
    <td class="tg-pb0m">0.1209</td>
  </tr>
  <tr>
    <td class="tg-pb0m">0.001</td>
    <td class="tg-pb0m">86.12</td>
    <td class="tg-pb0m">0.0019</td>
    <td class="tg-pb0m">90</td>
    <td class="tg-pb0m">0.8621</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="4">iris</td>
    <td class="tg-9wq8" rowspan="4">4</td>
    <td class="tg-9wq8" rowspan="4">depolarizing</td>
    <td class="tg-9wq8" rowspan="2">0.01</td>
    <td class="tg-pb0m">0.001</td>
    <td class="tg-pb0m">98.75</td>
    <td class="tg-pb0m">0.001</td>
    <td class="tg-pb0m">100</td>
    <td class="tg-pb0m">0.1529</td>
  </tr>
  <tr>
    <td class="tg-pb0m">0.01</td>
    <td class="tg-pb0m">92.5</td>
    <td class="tg-pb0m">0.001</td>
    <td class="tg-pb0m">100</td>
    <td class="tg-pb0m">5.0708</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">0.05</td>
    <td class="tg-pb0m">0.005</td>
    <td class="tg-pb0m">97.5</td>
    <td class="tg-pb0m">0.0011</td>
    <td class="tg-pb0m">100</td>
    <td class="tg-pb0m">0.2546</td>
  </tr>
  <tr>
    <td class="tg-pb0m">0.01</td>
    <td class="tg-pb0m">91.25</td>
    <td class="tg-pb0m">0.0011</td>
    <td class="tg-pb0m">100</td>
    <td class="tg-pb0m">3.2335</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="4">fashion</td>
    <td class="tg-9wq8" rowspan="4">8</td>
    <td class="tg-9wq8" rowspan="4">phase flip</td>
    <td class="tg-9wq8" rowspan="2">0.01</td>
    <td class="tg-pb0m">0.001</td>
    <td class="tg-pb0m">90.6</td>
    <td class="tg-pb0m">0.8519</td>
    <td class="tg-pb0m">97.4</td>
    <td class="tg-pb0m">8.921</td>
  </tr>
  <tr>
    <td class="tg-pb0m">0.01</td>
    <td class="tg-pb0m">54.4</td>
    <td class="tg-pb0m">0.7487</td>
    <td class="tg-pb0m">89.1</td>
    <td class="tg-pb0m">176.9451</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">0.05</td>
    <td class="tg-pb0m">0.001</td>
    <td class="tg-pb0m">90.6</td>
    <td class="tg-pb0m">0.8369</td>
    <td class="tg-pb0m">97.4</td>
    <td class="tg-pb0m">60.0099</td>
  </tr>
  <tr>
    <td class="tg-pb0m">0.005</td>
    <td class="tg-pb0m">71.9</td>
    <td class="tg-pb0m">0.7713</td>
    <td class="tg-pb0m">93.2</td>
    <td class="tg-pb0m">101.3314</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="4">mnist49</td>
    <td class="tg-9wq8" rowspan="4">8</td>
    <td class="tg-9wq8" rowspan="4">bit flip</td>
    <td class="tg-9wq8" rowspan="2">0.001</td>
    <td class="tg-pb0m">0.001</td>
    <td class="tg-pb0m">92.2</td>
    <td class="tg-pb0m">0.5289</td>
    <td class="tg-pb0m">93.7</td>
    <td class="tg-pb0m">10.6985</td>
  </tr>
  <tr>
    <td class="tg-pb0m">0.003</td>
    <td class="tg-pb0m">82.6</td>
    <td class="tg-pb0m">0.6346</td>
    <td class="tg-pb0m">88</td>
    <td class="tg-pb0m">50.3715</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">0.01</td>
    <td class="tg-pb0m">0.0001</td>
    <td class="tg-pb0m">97.1</td>
    <td class="tg-pb0m">0.7101</td>
    <td class="tg-pb0m">97.7</td>
    <td class="tg-pb0m">2.7965</td>
  </tr>
  <tr>
    <td class="tg-pb0m">0.001</td>
    <td class="tg-pb0m">91.9</td>
    <td class="tg-pb0m">0.5112</td>
    <td class="tg-pb0m">93.7</td>
    <td class="tg-pb0m">14.1884</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="4">tfi</td>
    <td class="tg-9wq8" rowspan="4">4</td>
    <td class="tg-9wq8" rowspan="4">mixed</td>
    <td class="tg-9wq8" rowspan="2">0.01</td>
    <td class="tg-9wq8">0.001</td>
    <td class="tg-pb0m">94.38</td>
    <td class="tg-pb0m">0.0025</td>
    <td class="tg-pb0m">100</td>
    <td class="tg-pb0m">0.7577</td>
  </tr>
  <tr>
    <td class="tg-9wq8">0.005</td>
    <td class="tg-pb0m">86.25</td>
    <td class="tg-pb0m">0.0034</td>
    <td class="tg-pb0m">100</td>
    <td class="tg-pb0m">1.8267</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">0.05</td>
    <td class="tg-9wq8">0.005</td>
    <td class="tg-pb0m">85.78</td>
    <td class="tg-pb0m">0.0034</td>
    <td class="tg-pb0m">100</td>
    <td class="tg-pb0m">1.7682</td>
  </tr>
  <tr>
    <td class="tg-9wq8">0.01</td>
    <td class="tg-pb0m">79.22</td>
    <td class="tg-pb0m">0.0042</td>
    <td class="tg-pb0m">100</td>
    <td class="tg-pb0m">2.6395</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="4">tfi</td>
    <td class="tg-9wq8" rowspan="4">8</td>
    <td class="tg-9wq8" rowspan="4">bit flip</td>
    <td class="tg-9wq8" rowspan="2">0.01</td>
    <td class="tg-9wq8">0.001</td>
    <td class="tg-pb0m">98.44</td>
    <td class="tg-pb0m">0.7339</td>
    <td class="tg-pb0m">100</td>
    <td class="tg-pb0m">1.3427</td>
  </tr>
  <tr>
    <td class="tg-9wq8">0.01</td>
    <td class="tg-pb0m">90.31</td>
    <td class="tg-pb0m">0.7591</td>
    <td class="tg-pb0m">100</td>
    <td class="tg-pb0m">4.4057</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">0.05</td>
    <td class="tg-pb0m">0.005</td>
    <td class="tg-pb0m">92.97</td>
    <td class="tg-pb0m">0.8704</td>
    <td class="tg-pb0m">100</td>
    <td class="tg-pb0m">4.8931</td>
  </tr>
  <tr>
    <td class="tg-9wq8">0.01</td>
    <td class="tg-pb0m">89.22</td>
    <td class="tg-pb0m">0.8339</td>
    <td class="tg-pb0m">100</td>
    <td class="tg-pb0m">6.5601</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="4">tfi_8</td>
    <td class="tg-c3ow" rowspan="4">8</td>
    <td class="tg-c3ow" rowspan="4">mixed</td>
    <td class="tg-c3ow" rowspan="2">0.01</td>
    <td class="tg-c3ow">0.001</td>
    <td class="tg-c3ow">98.59</td>
    <td class="tg-c3ow">5.5427</td>
    <td class="tg-c3ow">100</td>
    <td class="tg-c3ow">6.4524</td>
  </tr>
  <tr>
    <td class="tg-c3ow">0.01</td>
    <td class="tg-c3ow">90.62</td>
    <td class="tg-c3ow">5.2601</td>
    <td class="tg-c3ow">100</td>
    <td class="tg-c3ow">10.2166</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2">0.05</td>
    <td class="tg-c3ow">0.005</td>
    <td class="tg-c3ow">93.59</td>
    <td class="tg-c3ow">5.6429</td>
    <td class="tg-c3ow">100</td>
    <td class="tg-c3ow">9.4697</td>
  </tr>
  <tr>
    <td class="tg-c3ow">0.01</td>
    <td class="tg-c3ow">89.69</td>
    <td class="tg-c3ow">4.4898</td>
    <td class="tg-c3ow">100</td>
    <td class="tg-c3ow">10.1605</td>
  </tr>
  <tr>
    <td class="tg-9wq8">mnist01</td>
    <td class="tg-pb0m">8</td>
    <td class="tg-9wq8">phase flip</td>
    <td class="tg-9wq8">0.0005</td>
    <td class="tg-9wq8">0.0001</td>
    <td class="tg-9wq8">97.8</td>
    <td class="tg-9wq8">0.9794</td>
    <td class="tg-9wq8">98.5</td>
    <td class="tg-9wq8">5.8255</td>
  </tr>
  <tr>
    <td class="tg-nrix">mnist02</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">depolarizing</td>
    <td class="tg-nrix">0.0005</td>
    <td class="tg-nrix">0.005</td>
    <td class="tg-nrix">62.6</td>
    <td class="tg-nrix">30.294</td>
    <td class="tg-nrix">74.9</td>
    <td class="tg-nrix">107.9118</td>
  </tr>
  <tr>
    <td class="tg-nrix">mnist03</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">phase flip</td>
    <td class="tg-nrix">0.025</td>
    <td class="tg-nrix">0.003</td>
    <td class="tg-nrix">94.2</td>
    <td class="tg-nrix">0.7959</td>
    <td class="tg-nrix">97.4</td>
    <td class="tg-nrix">11.672</td>
  </tr>
  <tr>
    <td class="tg-nrix">mnist04</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">phase flip</td>
    <td class="tg-nrix">0.0005</td>
    <td class="tg-nrix">0.003</td>
    <td class="tg-nrix">98.3</td>
    <td class="tg-nrix">0.8077</td>
    <td class="tg-nrix">99.5</td>
    <td class="tg-nrix">2.8234</td>
  </tr>
  <tr>
    <td class="tg-nrix">mnist05</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-nrix">0.025</td>
    <td class="tg-nrix">0.001</td>
    <td class="tg-nrix">99.9</td>
    <td class="tg-nrix">5.8606</td>
    <td class="tg-nrix">100</td>
    <td class="tg-nrix">5.9724</td>
  </tr>
  <tr>
    <td class="tg-nrix">mnist06</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-nrix">0.01</td>
    <td class="tg-nrix">0.0003</td>
    <td class="tg-nrix">100</td>
    <td class="tg-nrix">5.3062</td>
    <td class="tg-nrix">100</td>
    <td class="tg-nrix">5.3063</td>
  </tr>
  <tr>
    <td class="tg-nrix">mnist07</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">bit flip</td>
    <td class="tg-nrix">0.005</td>
    <td class="tg-nrix">0.0003</td>
    <td class="tg-nrix">97.4</td>
    <td class="tg-nrix">1.1405</td>
    <td class="tg-nrix">97.8</td>
    <td class="tg-nrix">6.7732</td>
  </tr>
  <tr>
    <td class="tg-nrix">mnist08</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">phase flip</td>
    <td class="tg-nrix">0.01</td>
    <td class="tg-nrix">0.0003</td>
    <td class="tg-nrix">98.8</td>
    <td class="tg-nrix">0.9005</td>
    <td class="tg-nrix">99.6</td>
    <td class="tg-nrix">2.7806</td>
  </tr>
  <tr>
    <td class="tg-nrix">mnist09</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">phase flip</td>
    <td class="tg-nrix">0.01</td>
    <td class="tg-nrix">0.0005</td>
    <td class="tg-nrix">82.7</td>
    <td class="tg-nrix">0.8899</td>
    <td class="tg-nrix">87.4</td>
    <td class="tg-nrix">39.4768</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist12</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">phase flip</td>
    <td class="tg-8d8j">0.05</td>
    <td class="tg-8d8j">0.003</td>
    <td class="tg-8d8j">74.6</td>
    <td class="tg-8d8j">0.8202</td>
    <td class="tg-8d8j">77.9</td>
    <td class="tg-8d8j">61.6236</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist13</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">phase flip</td>
    <td class="tg-8d8j">0.075</td>
    <td class="tg-8d8j">0.0003</td>
    <td class="tg-8d8j">98.6</td>
    <td class="tg-8d8j">0.9784</td>
    <td class="tg-8d8j">98.8</td>
    <td class="tg-8d8j">4.2581</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist14</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">depolarizing</td>
    <td class="tg-8d8j">0.075</td>
    <td class="tg-8d8j">0.03</td>
    <td class="tg-8d8j">85</td>
    <td class="tg-8d8j">0.7144</td>
    <td class="tg-8d8j">87.2</td>
    <td class="tg-8d8j">44.0592</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist15</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">bit flip</td>
    <td class="tg-8d8j">0.0001</td>
    <td class="tg-8d8j">0.03</td>
    <td class="tg-8d8j">66.7</td>
    <td class="tg-8d8j">0.8398</td>
    <td class="tg-8d8j">66.8</td>
    <td class="tg-8d8j">92.4636</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist16</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">bit flip</td>
    <td class="tg-8d8j">0.05</td>
    <td class="tg-8d8j">0.0003</td>
    <td class="tg-8d8j">99.9</td>
    <td class="tg-8d8j">0.8424</td>
    <td class="tg-8d8j">99.9</td>
    <td class="tg-8d8j">1.1047</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist17</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">bit flip</td>
    <td class="tg-8d8j">0.05</td>
    <td class="tg-8d8j">0.01</td>
    <td class="tg-8d8j">80.3</td>
    <td class="tg-8d8j">0.8068</td>
    <td class="tg-8d8j">88.9</td>
    <td class="tg-8d8j">134.6892</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist18</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">phase flip</td>
    <td class="tg-8d8j">0.001</td>
    <td class="tg-8d8j">0.03</td>
    <td class="tg-8d8j">68.3</td>
    <td class="tg-8d8j">0.7612</td>
    <td class="tg-8d8j">68.3</td>
    <td class="tg-8d8j">81.6266</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist19</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-8d8j">0.01</td>
    <td class="tg-8d8j">0.075</td>
    <td class="tg-8d8j">55.6</td>
    <td class="tg-8d8j">4.4499</td>
    <td class="tg-8d8j">75.5</td>
    <td class="tg-8d8j">81.2824</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist23</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">phase flip</td>
    <td class="tg-8d8j">0.075</td>
    <td class="tg-8d8j">0.01</td>
    <td class="tg-8d8j">54</td>
    <td class="tg-8d8j">0.7597</td>
    <td class="tg-8d8j">58.1</td>
    <td class="tg-8d8j">106.9021</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist24</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">phase flip</td>
    <td class="tg-8d8j">0.005</td>
    <td class="tg-8d8j">0.01</td>
    <td class="tg-8d8j">43.7</td>
    <td class="tg-8d8j">0.7639</td>
    <td class="tg-8d8j">44.3</td>
    <td class="tg-8d8j">135.0514</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist25</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">phase flip</td>
    <td class="tg-8d8j">0.0005</td>
    <td class="tg-8d8j">0.003</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.7725</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.7725</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist26</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">depolarizing</td>
    <td class="tg-8d8j">0.075</td>
    <td class="tg-8d8j">0.0003</td>
    <td class="tg-8d8j">99.9</td>
    <td class="tg-8d8j">0.5375</td>
    <td class="tg-8d8j">99.9</td>
    <td class="tg-8d8j">0.7692</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist27</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">bit flip</td>
    <td class="tg-8d8j">0.001</td>
    <td class="tg-8d8j">0.0003</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">1.0876</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">1.0877</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist28</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">bit flip</td>
    <td class="tg-8d8j">0.001</td>
    <td class="tg-8d8j">0.0003</td>
    <td class="tg-8d8j">98.9</td>
    <td class="tg-8d8j">0.8244</td>
    <td class="tg-8d8j">99.4</td>
    <td class="tg-8d8j">2.628</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist29</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">phase flip</td>
    <td class="tg-8d8j">0.0001</td>
    <td class="tg-8d8j">0.003</td>
    <td class="tg-8d8j">99.9</td>
    <td class="tg-8d8j">0.7754</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.8467</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist34</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-8d8j">0.01</td>
    <td class="tg-8d8j">0.05</td>
    <td class="tg-8d8j">74.4</td>
    <td class="tg-8d8j">4.7773</td>
    <td class="tg-8d8j">97.6</td>
    <td class="tg-8d8j">28.4311</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist35</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-8d8j">0.025</td>
    <td class="tg-8d8j">0.075</td>
    <td class="tg-8d8j">50</td>
    <td class="tg-8d8j">4.3067</td>
    <td class="tg-8d8j">52.5</td>
    <td class="tg-8d8j">122.3814</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist36</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-8d8j">0.005</td>
    <td class="tg-8d8j">0.001</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">4.3917</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">4.3917</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist37</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">bit flip</td>
    <td class="tg-8d8j">0.05</td>
    <td class="tg-8d8j">0.001</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.7519</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.752</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist38</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-8d8j">0.025</td>
    <td class="tg-8d8j">0.0005</td>
    <td class="tg-8d8j">99.8</td>
    <td class="tg-8d8j">4.5474</td>
    <td class="tg-8d8j">99.8</td>
    <td class="tg-8d8j">5.0365</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist39</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">depolarizing</td>
    <td class="tg-8d8j">0.05</td>
    <td class="tg-8d8j">0.001</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.5051</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.5051</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist45</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-8d8j">0.01</td>
    <td class="tg-8d8j">0.003</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">4.0323</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">4.0323</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist46</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-8d8j">0.01</td>
    <td class="tg-8d8j">0.01</td>
    <td class="tg-8d8j">99</td>
    <td class="tg-8d8j">4.0782</td>
    <td class="tg-8d8j">99.8</td>
    <td class="tg-8d8j">5.1361</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist47</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">phase flip</td>
    <td class="tg-8d8j">0.0005</td>
    <td class="tg-8d8j">0.003</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.7727</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.7727</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist48</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-8d8j">0.05</td>
    <td class="tg-8d8j">0.03</td>
    <td class="tg-8d8j">45.6</td>
    <td class="tg-8d8j">5.9031</td>
    <td class="tg-8d8j">45.7</td>
    <td class="tg-8d8j">137.4534</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist49</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">bit flip</td>
    <td class="tg-8d8j">0.01</td>
    <td class="tg-8d8j">0.003</td>
    <td class="tg-8d8j">81.6</td>
    <td class="tg-8d8j">0.754</td>
    <td class="tg-8d8j">88</td>
    <td class="tg-8d8j">34.1436</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist56</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">bit flip</td>
    <td class="tg-8d8j">0.025</td>
    <td class="tg-8d8j">0.001</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.7607</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.7607</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist57</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">depolarizing</td>
    <td class="tg-8d8j">0.0001</td>
    <td class="tg-8d8j">0.0003</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.6417</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.6417</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist58</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">phase flip</td>
    <td class="tg-8d8j">0.05</td>
    <td class="tg-8d8j">0.01</td>
    <td class="tg-8d8j">98</td>
    <td class="tg-8d8j">1.0737</td>
    <td class="tg-8d8j">99.7</td>
    <td class="tg-8d8j">3.4832</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist59</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">bit flip</td>
    <td class="tg-8d8j">0.05</td>
    <td class="tg-8d8j">0.003</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.8439</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.8439</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist67</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-8d8j">0.005</td>
    <td class="tg-8d8j">0.03</td>
    <td class="tg-8d8j">95.8</td>
    <td class="tg-8d8j">4.4031</td>
    <td class="tg-8d8j">99.6</td>
    <td class="tg-8d8j">9.3411</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist68</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-8d8j">0.0001</td>
    <td class="tg-8d8j">0.075</td>
    <td class="tg-8d8j">50</td>
    <td class="tg-8d8j">4.2098</td>
    <td class="tg-8d8j">59.9</td>
    <td class="tg-8d8j">114.269</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist69</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-8d8j">0.05</td>
    <td class="tg-8d8j">0.0005</td>
    <td class="tg-8d8j">98.7</td>
    <td class="tg-8d8j">4.2002</td>
    <td class="tg-8d8j">99.4</td>
    <td class="tg-8d8j">6.5185</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist78</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-8d8j">0.025</td>
    <td class="tg-8d8j">0.05</td>
    <td class="tg-8d8j">90.2</td>
    <td class="tg-8d8j">4.4387</td>
    <td class="tg-8d8j">98.7</td>
    <td class="tg-8d8j">15.4179</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist79</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">bit flip</td>
    <td class="tg-8d8j">0.025</td>
    <td class="tg-8d8j">0.0003</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.848</td>
    <td class="tg-8d8j">100</td>
    <td class="tg-8d8j">0.848</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mnist89</td>
    <td class="tg-8d8j">8</td>
    <td class="tg-nrix">mixed</td>
    <td class="tg-8d8j">0.005</td>
    <td class="tg-8d8j">0.075</td>
    <td class="tg-8d8j">71.1</td>
    <td class="tg-8d8j">4.6005</td>
    <td class="tg-8d8j">94.8</td>
    <td class="tg-8d8j">37.9945</td>
  </tr>
</tbody>
</table>

## Global Robustness Verification

For *global-robustness*, we also add different levels of noise to each quantum model. We tested **49** quantum models on *VeriQR*, covering QCNN, Quantum Approximate Optimization Algorithms (QAOA), Variational Quantum Eigensolver (VQE) and other algorithms models, including: 

- The model $aci$, trained on the [Adult-Income dataset](https://archive.ics.uci.edu/dataset/2/adult) for income prediction. 
- The model $fct$, trained on a [dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) that contains credit transactions made by European cardholders, used to detect fraudulent credit card transactions.  
- The model $cr$, trained on the [dataset ](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) for classifying people described by a set of attributes as good or bad credit risks. 
- The models $ehc$ of 6, 8, 10, 12 qubits are obtained from the experiments in [the paper](https://www.science.org/doi/abs/10.1126/science.abb9811), used for calculating the binding energy of hydrogen chains. 
- The model named $qaoa$ is used for hardware grid problems in [the paper](https://www.nature.com/articles/s41567-020-01105-y).
- The models $iris$ and $fashion$ are same as the one in **Local-robustness Verification**. 

The full experimental results are shown in the following table: 

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">model</th>
    <th class="tg-c3ow">#qubits</th>
    <th class="tg-c3ow">noise</th>
    <th class="tg-c3ow">p </th>
    <th class="tg-c3ow">(ε, δ)</th>
    <th class="tg-c3ow">K*</th>
    <th class="tg-c3ow">robust</th>
    <th class="tg-c3ow">time (sec.)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" rowspan="4">aci</td>
    <td class="tg-c3ow" rowspan="4">8</td>
    <td class="tg-c3ow">bit flip</td>
    <td class="tg-c3ow">0.0001</td>
    <td class="tg-c3ow">(0.003, 0.0001)</td>
    <td class="tg-c3ow">0.99984</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">1.98</td>
  </tr>
  <tr>
    <td class="tg-c3ow">depolarizing</td>
    <td class="tg-c3ow">0.025</td>
    <td class="tg-c3ow">(0.03, 0.0005)</td>
    <td class="tg-c3ow">0.92412</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">3.08</td>
  </tr>
  <tr>
    <td class="tg-c3ow">phase flip</td>
    <td class="tg-c3ow">0.05</td>
    <td class="tg-c3ow">(0.05, 0.001)</td>
    <td class="tg-c3ow">0.79528</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">4.48</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mixed</td>
    <td class="tg-c3ow">0.005</td>
    <td class="tg-c3ow">(0.005, 0.005)</td>
    <td class="tg-c3ow">0.78436</td>
    <td class="tg-c3ow">YES</td>
    <td class="tg-c3ow">2.97</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="4">fct</td>
    <td class="tg-c3ow" rowspan="4">9</td>
    <td class="tg-c3ow">bit flip</td>
    <td class="tg-c3ow">0.05</td>
    <td class="tg-c3ow">(0.075, 0.003)</td>
    <td class="tg-c3ow">0.9</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">2.13</td>
  </tr>
  <tr>
    <td class="tg-c3ow">depolarizing</td>
    <td class="tg-c3ow">0.05</td>
    <td class="tg-c3ow">(0.0003, 0.0001)</td>
    <td class="tg-c3ow">0.84</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">2.07</td>
  </tr>
  <tr>
    <td class="tg-c3ow">phase flip</td>
    <td class="tg-c3ow">0.01</td>
    <td class="tg-c3ow">(0.01, 0.0075)</td>
    <td class="tg-c3ow">0.84</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">8.32</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mixed</td>
    <td class="tg-c3ow">0.05</td>
    <td class="tg-c3ow">(0.075, 0.0075)</td>
    <td class="tg-c3ow">0.84</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">3.63</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="4">cr</td>
    <td class="tg-c3ow" rowspan="4">9</td>
    <td class="tg-c3ow">bit flip</td>
    <td class="tg-c3ow">0.025</td>
    <td class="tg-c3ow">(0.01, 0.0005)</td>
    <td class="tg-c3ow">0.95</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">2.08</td>
  </tr>
  <tr>
    <td class="tg-c3ow">depolarizing</td>
    <td class="tg-c3ow">0.005</td>
    <td class="tg-c3ow">(0.075, 0.005)</td>
    <td class="tg-c3ow">0.94366</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">2.18</td>
  </tr>
  <tr>
    <td class="tg-c3ow">phase flip</td>
    <td class="tg-c3ow">0.025</td>
    <td class="tg-c3ow">(0.0003, 0.0001)</td>
    <td class="tg-c3ow">0.94366</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">3.92</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mixed</td>
    <td class="tg-c3ow">0.025</td>
    <td class="tg-c3ow">(0.0001, 0.0001)</td>
    <td class="tg-c3ow">0.94366</td>
    <td class="tg-c3ow">YES</td>
    <td class="tg-c3ow">2.2</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="4">ehc</td>
    <td class="tg-c3ow" rowspan="4">6</td>
    <td class="tg-c3ow">bit flip</td>
    <td class="tg-c3ow">0.05</td>
    <td class="tg-c3ow">(0.001, 0.0005)</td>
    <td class="tg-c3ow">0.89995</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">2.82</td>
  </tr>
  <tr>
    <td class="tg-c3ow">depolarizing</td>
    <td class="tg-c3ow">0.075</td>
    <td class="tg-c3ow">(0.001, 0.0001)</td>
    <td class="tg-c3ow">0.80916</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">2.67</td>
  </tr>
  <tr>
    <td class="tg-c3ow">phase flip</td>
    <td class="tg-c3ow">0.0001</td>
    <td class="tg-c3ow">(0.005, 0.003)</td>
    <td class="tg-c3ow">0.80899</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">2.69</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mixed</td>
    <td class="tg-c3ow">0.01</td>
    <td class="tg-c3ow">(0.0003, 0.0005)</td>
    <td class="tg-c3ow">0.80903</td>
    <td class="tg-c3ow">YES</td>
    <td class="tg-c3ow">2.79</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="4">ehc</td>
    <td class="tg-c3ow" rowspan="4">8</td>
    <td class="tg-c3ow">bit flip</td>
    <td class="tg-c3ow">0.0001</td>
    <td class="tg-c3ow">(0.0003, 0.0075)</td>
    <td class="tg-c3ow">0.99976</td>
    <td class="tg-c3ow">YES</td>
    <td class="tg-c3ow">5</td>
  </tr>
  <tr>
    <td class="tg-c3ow">depolarizing</td>
    <td class="tg-c3ow">0.05</td>
    <td class="tg-c3ow">(0.001, 0.0075)</td>
    <td class="tg-c3ow">0.93287</td>
    <td class="tg-c3ow">YES</td>
    <td class="tg-c3ow">5.6</td>
  </tr>
  <tr>
    <td class="tg-c3ow">phase flip</td>
    <td class="tg-c3ow">0.025</td>
    <td class="tg-c3ow">(0.075, 0.0003)</td>
    <td class="tg-c3ow">0.9327</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">5.43</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mixed</td>
    <td class="tg-c3ow">0.0005</td>
    <td class="tg-c3ow">(0.005, 0.005)</td>
    <td class="tg-c3ow">0.9322</td>
    <td class="tg-c3ow">YES</td>
    <td class="tg-c3ow">5.76</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="4">ehc</td>
    <td class="tg-c3ow" rowspan="4">10</td>
    <td class="tg-c3ow">bit flip</td>
    <td class="tg-c3ow">0.075</td>
    <td class="tg-c3ow">(0.05, 0.0003)</td>
    <td class="tg-c3ow">0.85264</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">24.7</td>
  </tr>
  <tr>
    <td class="tg-c3ow">depolarizing</td>
    <td class="tg-c3ow">0.0005</td>
    <td class="tg-c3ow">(0.03, 0.001)</td>
    <td class="tg-c3ow">0.85212</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">17.74</td>
  </tr>
  <tr>
    <td class="tg-c3ow">phase flip</td>
    <td class="tg-c3ow">0.01</td>
    <td class="tg-c3ow">(0.0003, 0.0075)</td>
    <td class="tg-c3ow">0.85058</td>
    <td class="tg-c3ow">YES</td>
    <td class="tg-c3ow">14.32</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mixed</td>
    <td class="tg-c3ow">0.0001</td>
    <td class="tg-c3ow">(0.005, 0.001)</td>
    <td class="tg-c3ow">0.85027</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">22.34</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="4">ehc</td>
    <td class="tg-c3ow" rowspan="4">12</td>
    <td class="tg-c3ow">bit flip</td>
    <td class="tg-c3ow">0.005</td>
    <td class="tg-c3ow">(0.0005, 0.0003)</td>
    <td class="tg-c3ow">0.98966</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">25.73</td>
  </tr>
  <tr>
    <td class="tg-c3ow">depolarizing</td>
    <td class="tg-c3ow">0.0005</td>
    <td class="tg-c3ow">(0.0001, 0.005)</td>
    <td class="tg-c3ow">0.99926</td>
    <td class="tg-c3ow">YES</td>
    <td class="tg-c3ow">21.22</td>
  </tr>
  <tr>
    <td class="tg-c3ow">phase flip</td>
    <td class="tg-c3ow">0.075</td>
    <td class="tg-c3ow">(0.001, 0.0075)</td>
    <td class="tg-c3ow">0.99883</td>
    <td class="tg-c3ow">YES</td>
    <td class="tg-c3ow">206.08</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mixed</td>
    <td class="tg-c3ow">0.001</td>
    <td class="tg-c3ow">(0.01, 0.0001)</td>
    <td class="tg-c3ow">0.99984</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">45.53</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="4">qaoa</td>
    <td class="tg-c3ow" rowspan="4">10</td>
    <td class="tg-c3ow">bit flip</td>
    <td class="tg-c3ow">0.005</td>
    <td class="tg-c3ow">(0.05, 0.0005)</td>
    <td class="tg-c3ow">0.98497</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">3.98</td>
  </tr>
  <tr>
    <td class="tg-c3ow">depolarizing</td>
    <td class="tg-c3ow">0.0001</td>
    <td class="tg-c3ow">(0.01, 0.003)</td>
    <td class="tg-c3ow">0.9847</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">3.89</td>
  </tr>
  <tr>
    <td class="tg-c3ow">phase flip</td>
    <td class="tg-c3ow">0.005</td>
    <td class="tg-c3ow">(0.075, 0.0075)</td>
    <td class="tg-c3ow">0.97097</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">4.28</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mixed</td>
    <td class="tg-c3ow">0.001</td>
    <td class="tg-c3ow">(0.03, 0.0075)</td>
    <td class="tg-c3ow">0.96874</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">4.33</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="4">fashion</td>
    <td class="tg-c3ow" rowspan="4">8</td>
    <td class="tg-c3ow">bit flip</td>
    <td class="tg-c3ow">0.005</td>
    <td class="tg-c3ow">(0.075, 0.005)</td>
    <td class="tg-c3ow">0.98987</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">1.8</td>
  </tr>
  <tr>
    <td class="tg-c3ow">depolarizing</td>
    <td class="tg-c3ow">0.025</td>
    <td class="tg-c3ow">(0.03, 0.003)</td>
    <td class="tg-c3ow">0.95307</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">1.9</td>
  </tr>
  <tr>
    <td class="tg-c3ow">phase flip</td>
    <td class="tg-c3ow">0.025</td>
    <td class="tg-c3ow">(0.005, 0.0003)</td>
    <td class="tg-c3ow">0.93769</td>
    <td class="tg-c3ow">NO</td>
    <td class="tg-c3ow">1.94</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mixed</td>
    <td class="tg-c3ow">0.075</td>
    <td class="tg-c3ow">(0.0005, 0.0075)</td>
    <td class="tg-c3ow">0.8326</td>
    <td class="tg-c3ow">YES</td>
    <td class="tg-c3ow">1.94</td>
  </tr>
    <tr>
    <td class="tg-nrix" rowspan="4">iris</td>
    <td class="tg-nrix" rowspan="4">4</td>
    <td class="tg-8d8j">bit flip</td>
    <td class="tg-8d8j">0.005</td>
    <td class="tg-8d8j">(0.003, 0.0001)</td>
    <td class="tg-8d8j">0.98622</td>
    <td class="tg-8d8j">NO</td>
    <td class="tg-8d8j">1.35</td>
  </tr>
  <tr>
    <td class="tg-8d8j">depolarizing</td>
    <td class="tg-8d8j">0.005</td>
    <td class="tg-8d8j">(0.03, 0.0075)</td>
    <td class="tg-8d8j">0.9673</td>
    <td class="tg-8d8j">NO</td>
    <td class="tg-8d8j">1.11</td>
  </tr>
  <tr>
    <td class="tg-8d8j">phase flip</td>
    <td class="tg-8d8j">0.0001</td>
    <td class="tg-8d8j">(0.005, 0.005)</td>
    <td class="tg-8d8j">0.96935</td>
    <td class="tg-8d8j">YES</td>
    <td class="tg-8d8j">1.2</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mixed</td>
    <td class="tg-8d8j">0.0001</td>
    <td class="tg-8d8j">(0.03, 0.005)</td>
    <td class="tg-8d8j">0.96811</td>
    <td class="tg-8d8j">NO</td>
    <td class="tg-8d8j">1.22</td>
  </tr>
</tbody>
</table>

