from EnvQ import EnvQ
import numpy as np


class IterativePolicyEvaluation:

    def __init__(self, env: EnvQ, theta=0.0001):
        self.theta = theta
        states = list(range(env.max_length))
        init_vals = [-np.inf] * env.max_length
        self.value_function = dict(zip(states, init_vals))

    def evaluate(self):
        delta = np.inf
        while delta > self.theta:
            for state in self.value_function:
                pass
