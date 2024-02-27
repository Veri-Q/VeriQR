import numpy as np
from numpy import load

data = load("./model_and_data/binary_cav.npz")
kraus = data["kraus"]
np.savez('./kraus/kraus_1qubit.npz', kraus=kraus)

