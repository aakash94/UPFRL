import time
import copy
import random
import numpy as np

from matplotlib import pyplot as plt
from collections import defaultdict

from EnvQ import EnvQ
from Policies import policy_iteration, value_iteration
from IterativePolicyEvaluation import IterativePolicyEvaluation

def alpha_function(timestep = 0, a = 10^5, b = 10^5):
    return a/(timestep +b)
    
class TD():
    def __init__(self, env: EnvQ):
        self.env = env

    def evaluate (self, policy, env, num_episodes, alpha_function = alpha_function , gamma = .9):
        V = defaultdict(float)
        number_of_actions = len(self.env.actions)
        t = 0
        for i_episode in range(1, num_episodes+1):
            state=self.env.reset()
            while True:
                delta = 0
                action = np.choice(self.env.actions, 1, p=policy[state])
                next_state, reward, done, _ = self.env.step(action)
                alpha = alpha_function(t)
                V[state] += alpha * (reward + gamma * V[next_state] - V[state])
                if done:
                    break
        return V


if __name__ == '__main__':
    td = TD()