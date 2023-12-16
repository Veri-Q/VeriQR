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

