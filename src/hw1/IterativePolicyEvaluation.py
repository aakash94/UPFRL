from EnvQ import EnvQ
import numpy as np


class IterativePolicyEvaluation:

    def __init__(self, env: EnvQ, theta=0.0001):
        self.theta = theta
        states = list(range(env.max_length))
        init_vals = [-np.inf] * env.max_length
        self.value_function = dict(zip(states, init_vals))

    def evaluate(self, pi, gamma, V):
        while True:
            delta = 0
            for s in env.S:
                v = V[s]
                bellman_update(env, V, pi, s, gamma)
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        return V
