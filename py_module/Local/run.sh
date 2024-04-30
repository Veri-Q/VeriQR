# verification:
# python local_verif.py qasmfile datafile eps veri_num state_flag
python local_verif.py ./model_and_data/qubit_cav.npz 0.001 5 mixed
python local_verif.py ./model_and_data/iris.qasm ./model_and_data/iris_data.npz 0.001 1 mixed false depolarizing 0.01
python local_verif.py ./model_and_data/fashion8.qasm ./model_and_data/fashion8_data.npz 0.001 1 pure false bit_flip 0.01
python local_verif.py ./model_and_data/mnist13.qasm ./model_and_data/mnist13_data.npz 0.001 1 pure false phase_flip 0.001
python local_verif.py ./model_and_data/tfi4.qasm ./model_and_data/tfi4_data.npz 0.001 1 pure false mixed 0.01
python local_verif.py ./model_and_data/tfi8.qasm ./model_and_data/tfi8_data.npz 0.001 1 pure false bit_flip 0.01

# evaluation:
# evaluate()
# python evaluate.py qasmfile datafile state_flag
python evaluate.py ./model_and_data/iris.qasm ./model_and_data/iris_data.npz mixed
python evaluate.py ./model_and_data/tfi4.qasm ./model_and_data/tfi4_data.npz pure
python evaluate.py ./model_and_data/tfi8.qasm ./model_and_data/tfi8_data.npz pure
python evaluate.py ./model_and_data/fashion8.qasm ./model_and_data/fashion8_data.npz pure
python evaluate.py ./model_and_data/mnist13.qasm ./model_and_data/mnist13_data.npz pure
# for evaluate_mnist()
# python evaluate.py digits
python evaluate.py 13

# evaluation for AT:
# python evaluate_for_AT.py qasmfile datafile state_flag eps (noise_setting)
python evaluate_for_AT.py ./model_and_data/iris.qasm ./model_and_data/iris_data.npz mixed 0.001 depolarizing 0.01
