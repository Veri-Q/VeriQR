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

## Local Robustness Verification

VeriQR provides several labeled datasets, all of which are encoded into quantum data by different quantum encoders and trained on Mindspore or Tensorflow Quantum platforms to generate QML models for verification. In total, we verify the local-robustness of 50 quantum classifiers, including: 

-  The model $qubit$, used to identify the region of a quantum state in the X-Z plane of the Bloch sphere. 
- The model $iris$, trained on the [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) that is encoded into mixed states of 4 qubits via Instantaneous Quantum Polynomial (IQP) encoding, is used for classifying irises of different subgenera. 
- The models $mnist$, trained on the well-known MNIST dataset that is encoded into pure states of 8 qubits using amplitude encoding, is used for classifying handwritten digits, which can be any two of numbers ranging from 0 to 9. 
- The model $fashion$, trained on the well-known Fashion MNIST dataset that is encoded into pure states of 8 qubits via amplitude encoding, is used to classify images of two fashion products, T-shirts and ankle boots. 
-  The models $tfi$, trained on the data points of 4 qubits and of 8 qubits in the [TFI\_chain dataset](https://tensorflow.google.cn/quantum/api_docs/python/tfq/datasets/tfi_chain) respectively using the approach in [the paper](https://www.scipost.org/SciPostPhysLectNotes.61), are used for identifying the wavefunction at different phases in a quantum many-body system. 

We conducted numerous experiments on different circuits for each model as outlined in the [local_results](https://github.com/Veri-Q/VeriQR/blob/main/py_module/Local/results/local_results.csv) table: 

- The noiseless ideal QML model with quantum circuit $c_0$; 
- Circuit $c_1$ created by introducing random noise at various random points in circuit $c_0$ to simulate noise effects on NISQ devices; 
- Circuit $c_2$ modified by adding specific noise with a noise level $0 \leq p \leq 1$ of four types: *depolarizing*, *phase flip*, *bit flip*, and *mixed* (a combination of the three) noise (referred to as "noise_p" below $c_2$), applied to each qubit after the random noise manipulation on circuit $c_1$. 

Where $RA$ indicates the robust accuracy of classifiers, and $VT$ is the verification time in seconds. 

Here are some of the experimental results: 

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">#Qubits</th>
    <th rowspan="2">ε</th>
    <th rowspan="2">Circuit</th>
    <th rowspan="2">Noise Setting (noise_p)</th>
    <th colspan="2">Rough Verif</th>
    <th colspan="2">Accurate Verif</th>
  </tr>
  <tr>
    <td>RA (%)</td>
    <td>VT (sec.)</td>
    <td>RA (%)</td>
    <td>VT (sec.)</td>
  </tr>
</thead>
<tbody>
    <tr>
        <td rowspan="4">qubit</td>
        <td rowspan="4">1</td>
        <td rowspan="4">0.001</td>
        <td>c_0</td>
        <td>noiseless</td>
        <td>88.12</td>
        <td>0.0038</td>
        <td>90</td>
        <td>2.4226</td>
    </tr>
    <tr>
        <td>c_1</td>
        <td>random</td>
        <td>88.12</td>
        <td>0.0039</td>
        <td>90</td>
        <td>2.4623</td>
    </tr>
    <tr>
        <td>c_2</td>
        <td>depolarizing_0.001</td>
        <td>88.00</td>
        <td>0.0038</td>
        <td>90</td>
        <td>2.4873</td>
    </tr>
    <tr>
        <td>c_2</td>
        <td>depolarizing_0.005</td>
        <td>87.62</td>
        <td>0.0053</td>
        <td>90</td>
        <td>2.7140</td>
    </tr>
    <tr>
        <td rowspan="4">iris</td>
        <td rowspan="4">4</td>
        <td rowspan="4">0.005</td>
        <td>c_0</td>
        <td>noiseless</td>
        <td>98.75</td>
        <td>0.0013</td>
        <td>100</td>
        <td>0.4924</td>
    </tr>
    <tr>
        <td>c_1</td>
        <td>random</td>
        <td>97.50</td>
        <td>0.0009</td>
        <td>100</td>
        <td>0.8876</td>
    </tr>
    <tr>
        <td>c_2</td>
        <td>mixed_0.01</td>
        <td>97.50</td>
        <td>0.0019</td>
        <td>100</td>
        <td>0.8808</td>
    </tr>
    <tr>
        <td>c_2</td>
        <td>mixed_0.05</td>
        <td>96.25</td>
        <td>0.0021</td>
        <td>100</td>
        <td>3.1675</td>
    </tr>
    <tr>
        <td rowspan="4">tfi</td>
        <td rowspan="4">4</td>
        <td rowspan="4">0.005</td>
        <td>c_0</td>
        <td>noiseless</td>
        <td>86.41</td>
        <td>0.0039</td>
        <td>100</td>
        <td>6.5220</td>
    </tr>
    <tr>
        <td>c_1</td>
        <td>random</td>
        <td>85.94</td>
        <td>0.0038</td>
        <td>100</td>
        <td>6.6438</td>
    </tr>
    <tr>
        <td>c_2</td>
        <td>mixed_0.01</td>
        <td>85.78</td>
        <td>0.0061</td>
        <td>100</td>
        <td>6.7117</td>
    </tr>
    <tr>
        <td>c_2</td>
        <td>mixed_0.05</td>
        <td>85.16</td>
        <td>0.0063</td>
        <td>100</td>
        <td>7.0374</td>
    </tr>
    <tr>
        <td rowspan="4">tfi</td>
        <td rowspan="4">8</td>
        <td rowspan="4">0.005</td>
        <td>c_0</td>
        <td>noiseless</td>
        <td>98.44</td>
        <td>0.0372</td>
        <td>100</td>
        <td>2.3004</td>
    </tr>
    <tr>
        <td>c_1</td>
        <td>random</td>
        <td>96.56</td>
        <td>0.1061</td>
        <td>100</td>
        <td>3.9492</td>
    </tr>
    <tr>
        <td>c_2</td>
        <td>bit-flip_0.01</td>
        <td>96.56</td>
        <td>37.0965</td>
        <td>100</td>
        <td>42.1246</td>
    </tr>
    <tr>
        <td>c_2</td>
        <td>bit-flip_0.05</td>
        <td>95.94</td>
        <td>32.7195</td>
        <td>100</td>
        <td>38.8139</td>
    </tr>
    <tr>
        <td rowspan="4">fashion</td>
        <td rowspan="4">8</td>
        <td rowspan="4">0.001</td>
        <td>c_0</td>
        <td>noiseless</td>
        <td>90.60</td>
        <td>0.0420</td>
        <td>97.40</td>
        <td>25.3777</td>
    </tr>
    <tr>
        <td>c_1</td>
        <td>random</td>
        <td>90.30</td>
        <td>0.0934</td>
        <td>97.30</td>
        <td>27.4964</td>
    </tr>
    <tr>
        <td>c_2</td>
        <td>bit-flip_0.01</td>
        <td>89.90</td>
        <td>15.6579</td>
        <td>97.20</td>
        <td>42.1063</td>
    </tr>
    <tr>
        <td>c_2</td>
        <td>bit-flip_0.05</td>
        <td>87.60</td>
        <td>14.0342</td>
        <td>96.70</td>
        <td>48.5805</td>
    </tr>
    <tr>
        <td rowspan="4">mnist (1&3)</td>
        <td rowspan="4">8</td>
        <td rowspan="4">0.003</td>
        <td>c_0</td>
        <td>noiseless</td>
        <td>93.80</td>
        <td>0.0543</td>
        <td>96.00</td>
        <td>18.5063</td>
    </tr>
    <tr>
        <td>c_1</td>
        <td>random</td>
        <td>92.60</td>
        <td>0.0785</td>
        <td>95.70</td>
        <td>23.2905</td>
    </tr>
    <tr>
        <td>c_2</td>
        <td>phase-flip_0.001</td>
        <td>92.60</td>
        <td>12.9728</td>
        <td>95.70</td>
        <td>36.2348</td>
    </tr>
    <tr>
        <td>c_2</td>
        <td>phase-flip_0.01</td>
        <td>92.60</td>
        <td>11.6704</td>
        <td>95.70</td>
        <td>33.7894</td>
    </tr>
</tbody>
</table>

### Adversarial Training for Improving Robustness

*VeriQR* empowers users with adversarial training capabilities, an extension of traditional machine learning. When the $\epsilon$-local robustness of $\rho$ with label $l$ is compromised, our robustness verification algorithms embedded in *VeriQR* automatically generate an adversarial example $\sigma$. By incorporating a number of states and their ground truth labels $(\sigma, l)$ into the training dataset, we retrained several QML model to enhance their local robustness against the adversarial examples. The results are shown as followed, demonstrating the effectiveness of adversarial training in boosting the local robustness of QMLs, where $NRN$ represents the number of non-robust states in the original dataset  for each model. In particular, since most of our experimental models achieved 100% RA, we use "`-`" entries here to indicate that these models do not need adversarial training after initial validation. 

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">#Qubits</th>
    <th rowspan="2">Circuit</th>
    <th rowspan="2">Noise&nbsp;&nbsp;&nbsp;Setting<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(noise_p)</th>
    <th rowspan="2">ε</th>
    <th rowspan="2">Traning</th>
    <th colspan="3">Rough Verif</th>
    <th colspan="3">Accurate Verif</th>
  </tr>
  <tr>
    <th>NRN</th>
    <th>RA (%)</th>
    <th>VT (sec.)</th>
    <th>NRN</th>
    <th>RA (%)</th>
    <th>VT (sec.)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="24">fashion</td>
    <td rowspan="24">8</td>
    <td rowspan="4">c_0</td>
    <td rowspan="4">noiseless</td>
    <td rowspan="2">0.001</td>
    <td>before</td>
    <td>94</td>
    <td>90.60 </td>
    <td>0.0471 </td>
    <td>26</td>
    <td>97.40 </td>
    <td>19.9476 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>90</td>
    <td>90.06 </td>
    <td>0.0450 </td>
    <td>15</td>
    <td>98.54 </td>
    <td>27.0747 </td>
  </tr>
  <tr>
    <td rowspan="2">0.005</td>
    <td>before</td>
    <td>187</td>
    <td>81.30 </td>
    <td>0.0207 </td>
    <td>44</td>
    <td>95.60 </td>
    <td>38.1464 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>56</td>
    <td>93.10 </td>
    <td>0.0331 </td>
    <td>27</td>
    <td>97.41 </td>
    <td>17.0312 </td>
  </tr>
  <tr>
    <td rowspan="4">c_1</td>
    <td rowspan="4">random</td>
    <td rowspan="2">0.001</td>
    <td>before</td>
    <td>111</td>
    <td>88.90 </td>
    <td>1.5387 </td>
    <td>30</td>
    <td>97.00 </td>
    <td>24.2540 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>33</td>
    <td>95.92 </td>
    <td>0.0310 </td>
    <td>10</td>
    <td>99.03 </td>
    <td>10.5288 </td>
  </tr>
  <tr>
    <td rowspan="2">0.005</td>
    <td>before</td>
    <td>215</td>
    <td>78.50 </td>
    <td>1.5210 </td>
    <td>50</td>
    <td>95.00 </td>
    <td>54.8998 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>70</td>
    <td>91.62 </td>
    <td>0.0329 </td>
    <td>38</td>
    <td>96.38 </td>
    <td>21.5003 </td>
  </tr>
  <tr>
    <td rowspan="4">c_2</td>
    <td rowspan="4">bit-flip_0.01</td>
    <td rowspan="2">0.001</td>
    <td>before</td>
    <td>120</td>
    <td>88.00 </td>
    <td>19.0693 </td>
    <td>24</td>
    <td>97.60 </td>
    <td>44.5314 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>143</td>
    <td>85.35 </td>
    <td>0.0380 </td>
    <td>13</td>
    <td>98.73 </td>
    <td>41.7078 </td>
  </tr>
  <tr>
    <td rowspan="2">0.005</td>
    <td>before</td>
    <td>263</td>
    <td>73.70 </td>
    <td>25.1982 </td>
    <td>41</td>
    <td>95.90 </td>
    <td>81.4979 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>88</td>
    <td>90.01 </td>
    <td>0.0316 </td>
    <td>34</td>
    <td>96.73 </td>
    <td>26.1886 </td>
  </tr>
  <tr>
    <td rowspan="4">c_2</td>
    <td rowspan="4">depolarizing_0.001</td>
    <td rowspan="2">0.001</td>
    <td>before</td>
    <td>121</td>
    <td>87.90 </td>
    <td>18.4088 </td>
    <td>24</td>
    <td>97.60 </td>
    <td>50.5345 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>135</td>
    <td>86.04 </td>
    <td>0.0355 </td>
    <td>12</td>
    <td>98.83 </td>
    <td>39.0603 </td>
  </tr>
  <tr>
    <td rowspan="2">0.005</td>
    <td>before</td>
    <td>266</td>
    <td>73.40 </td>
    <td>22.2503 </td>
    <td>41</td>
    <td>95.90 </td>
    <td>93.6130 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>88</td>
    <td>90.01 </td>
    <td>0.0341 </td>
    <td>34</td>
    <td>96.73 </td>
    <td>24.0009 </td>
  </tr>
  <tr>
    <td rowspan="4">c_2</td>
    <td rowspan="4">phase-flip_0.001</td>
    <td rowspan="2">0.001</td>
    <td>before</td>
    <td>120</td>
    <td>88.00 </td>
    <td>21.1309 </td>
    <td>24</td>
    <td>97.60 </td>
    <td>52.8487 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>134</td>
    <td>86.13 </td>
    <td>0.0237 </td>
    <td>12</td>
    <td>98.83 </td>
    <td>29.8855 </td>
  </tr>
  <tr>
    <td rowspan="2">0.005</td>
    <td>before</td>
    <td>262</td>
    <td>73.80 </td>
    <td>21.1185 </td>
    <td>41</td>
    <td>95.90 </td>
    <td>78.1674 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>67</td>
    <td>92.12 </td>
    <td>0.0224 </td>
    <td>25</td>
    <td>97.60 </td>
    <td>15.3171 </td>
  </tr>
  <tr>
    <td rowspan="4">c_2</td>
    <td rowspan="4">mixed_0.01</td>
    <td rowspan="2">0.001</td>
    <td>before</td>
    <td>120</td>
    <td>88.00 </td>
    <td>171.4938 </td>
    <td>24</td>
    <td>97.60 </td>
    <td>203.8980 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>135</td>
    <td>86.04 </td>
    <td>0.0233 </td>
    <td>12</td>
    <td>98.83 </td>
    <td>29.9190 </td>
  </tr>
  <tr>
    <td rowspan="2">0.005</td>
    <td>before</td>
    <td>263</td>
    <td>73.70 </td>
    <td>174.0630 </td>
    <td>41</td>
    <td>95.90 </td>
    <td>246.1607 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>87</td>
    <td>90.11 </td>
    <td>0.0221 </td>
    <td>35</td>
    <td>96.64 </td>
    <td>19.6066 </td>
  </tr>
  <tr>
    <td rowspan="24">iris</td>
    <td rowspan="24">4</td>
    <td rowspan="4">c_0</td>
    <td rowspan="4">noiseless</td>
    <td rowspan="2">0.01</td>
    <td>before</td>
    <td>5</td>
    <td>93.75 </td>
    <td>0.0020 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>9.2720 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">0.05</td>
    <td>before</td>
    <td>42</td>
    <td>47.50 </td>
    <td>0.0009 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>137.7165 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="4">c_1</td>
    <td rowspan="4">random</td>
    <td rowspan="2">0.01</td>
    <td>before</td>
    <td>20</td>
    <td>75.00 </td>
    <td>0.0011 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>46.0594 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">0.05</td>
    <td>before</td>
    <td>47</td>
    <td>41.25 </td>
    <td>0.0011 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>208.1525 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="4">c_2</td>
    <td rowspan="4">bit-flip_0.005</td>
    <td rowspan="2">0.01</td>
    <td>before</td>
    <td>20</td>
    <td>75.00 </td>
    <td>0.0028 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>46.2143 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">0.05</td>
    <td>before</td>
    <td>47</td>
    <td>41.25 </td>
    <td>0.0031 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>208.8100 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="4">c_2</td>
    <td rowspan="4">depolarizing_0.0005</td>
    <td rowspan="2">0.01</td>
    <td>before</td>
    <td>20</td>
    <td>75.00 </td>
    <td>0.0028 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>46.0320 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">0.05</td>
    <td>before</td>
    <td>47</td>
    <td>41.25 </td>
    <td>0.0033 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>208.6576 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="4">c_2</td>
    <td rowspan="4">phase-flip_0.005</td>
    <td rowspan="2">0.01</td>
    <td>before</td>
    <td>20</td>
    <td>75.00 </td>
    <td>0.0029 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>46.0522 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">0.05</td>
    <td>before</td>
    <td>47</td>
    <td>41.25 </td>
    <td>0.0031 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>208.6859 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="4">c_2</td>
    <td rowspan="4">mixed_0.0005</td>
    <td rowspan="2">0.01</td>
    <td>before</td>
    <td>20</td>
    <td>75.00 </td>
    <td>0.0050 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>45.7089 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">0.05</td>
    <td>before</td>
    <td>47</td>
    <td>41.25 </td>
    <td>0.0055 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>211.5529 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="22">tfi</td>
    <td rowspan="22">4</td>
    <td rowspan="2">c_0</td>
    <td rowspan="2">noiseless</td>
    <td rowspan="2">0.1</td>
    <td>before</td>
    <td>558</td>
    <td>12.81 </td>
    <td>0.0036 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>42.4537 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="4">c_1</td>
    <td rowspan="4">random</td>
    <td rowspan="2">0.05</td>
    <td>before</td>
    <td>333</td>
    <td>47.97 </td>
    <td>0.0038 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>24.5601 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">0.1</td>
    <td>before</td>
    <td>551</td>
    <td>13.91 </td>
    <td>0.0037 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>41.0691 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="4">c_2</td>
    <td rowspan="4">bit-flip_0.02</td>
    <td rowspan="2">0.05</td>
    <td>before</td>
    <td>354</td>
    <td>44.69 </td>
    <td>0.0038 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>26.0638 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">0.1</td>
    <td>before</td>
    <td>571</td>
    <td>10.78 </td>
    <td>0.0039 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>42.6273 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="4">c_2</td>
    <td rowspan="4">depolarizing_0.005</td>
    <td rowspan="2">0.05</td>
    <td>before</td>
    <td>335</td>
    <td>47.66 </td>
    <td>0.0038 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>24.7343 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">0.1</td>
    <td>before</td>
    <td>553</td>
    <td>13.59 </td>
    <td>0.0039 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>41.2738 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="4">c_2</td>
    <td rowspan="4">phase-flip_0.0001</td>
    <td rowspan="2">0.05</td>
    <td>before</td>
    <td>333</td>
    <td>47.97 </td>
    <td>0.0039 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>24.5491 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">0.1</td>
    <td>before</td>
    <td>551</td>
    <td>13.91 </td>
    <td>0.0039 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>41.0684 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="4">c_2</td>
    <td rowspan="4">mixed_0.01</td>
    <td rowspan="2">0.05</td>
    <td>before</td>
    <td>338</td>
    <td>47.19 </td>
    <td>0.0041 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>25.0071 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">0.1</td>
    <td>before</td>
    <td>555</td>
    <td>13.28 </td>
    <td>0.0043 </td>
    <td>0</td>
    <td>100.00 </td>
    <td>41.4739 </td>
  </tr>
  <tr>
    <td>after</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## Global Robustness Verification

For *global-robustness*, we also add different levels of noise to each quantum model. We tested 12 QML models on *VeriQR*, covering QCNN, Quantum Approximate Optimization Algorithms (QAOA), Variational Quantum Eigensolver (VQE) and other algorithms, including: 

- The model $aci$, trained on the [Adult-Income dataset](https://archive.ics.uci.edu/dataset/2/adult) for income prediction. 
- The model $fct$, trained on a [dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) that contains credit transactions made by European cardholders, used to detect fraudulent credit card transactions.  
- The model $cr$, trained on the [dataset ](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) for classifying people described by a set of attributes as good or bad credit risks. 
- The models $ehc$ of 6, 8, 10, 12 qubits are obtained from the experiments in [the paper](https://www.science.org/doi/abs/10.1126/science.abb9811), used for calculating the binding energy of hydrogen chains. 
- The model named $qaoa$ is used for hardware grid problems in [the paper](https://www.nature.com/articles/s41567-020-01105-y).
- The models $iris$ and $fashion$ are same as the one in **Local-robustness Verification**. 

The full experimental results are shown in the [global_results](https://github.com/Veri-Q/VeriQR/blob/main/py_module/Global/results/global_results.csv) table. 

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
