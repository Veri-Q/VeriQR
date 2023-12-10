from VeriQ import RobustnessVerifier, PureRobustnessVerifier
from numpy import load

data_file = 'mnist_cav.npz'
eps = 1e-4

verifier = PureRobustnessVerifier

DATA = load(data_file)

ac_temp, time_temp = verifier(
    DATA['kraus'],
    DATA['O'],
    DATA['data'],
    DATA['label'],
    eps,
    ADVERSARY_EXAMPLE = True)

print('Adversary examples have been written into \'adversary_example_*.pdf\'')
