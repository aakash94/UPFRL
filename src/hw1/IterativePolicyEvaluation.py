from EnvQ import EnvQ
import numpy as np
from matplotlib import pyplot as plt


DISCOUNT_FACTOR = 0.9

class IterativePolicyEvaluation:

    def __init__(self, env: EnvQ):
        self.env = env

    def evaluate(self, policy, gamma=DISCOUNT_FACTOR, theta=1e-8):
        V = np.zeros(self.env.max_length)
        while True:
            delta = 0
            for s in range(self.env.max_length):
                Vs = 0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in self.env.transition[s][a]:
                        Vs += action_prob * prob * (reward + gamma * V[next_state])
                delta = max(delta, np.abs(V[s] - Vs))
                V[s] = Vs
            if delta < theta:
                break
        return V

    def plot_value_function(self, V):
        plt.bar(range(len(V)), V)
        plt.show()
