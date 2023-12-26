import jax.numpy as jnp


class Adam():
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, alpha=1e-2, sign=1):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iteration = 1
        self.m = 0
        self.v = 0
        self.alpha = alpha
        self.sign = sign

    def __call__(self, gradients):
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients * gradients)

        mhat = self.m / (1 - self.beta1 ** self.iteration)
        vhat = self.v / (1 - self.beta2 ** self.iteration)

        self.iteration += 1

        return self.sign * self.alpha * (mhat / (jnp.sqrt(vhat) + self.epsilon))
