import time
import copy
import random
import numpy as np

from matplotlib import pyplot as plt
from collections import defaultdict

from EnvQ import EnvQ
from Policies import get_lazy_policy, get_aggressive_policy
from IterativePolicyEvaluation import IterativePolicyEvaluation

SEED = 4

np.random.seed(SEED)


def alpha_function(timestep=0, a=10 ^ 5, b=10 ^ 5):
    return a / (timestep + b)


class TD():

    def __init__(self, env: EnvQ):
        self.env = env

    def evaluate(self, policy, alpha_function=alpha_function, gamma=.9):
        V = defaultdict(float)

        state = self.env.reset()
        policy_fun = policy()
        done = False
        timestep = 0
        while not done:
            action = np.random.choice(self.env.actions, 1, p=policy_fun[state])
            next_state, reward, done, _ = self.env.step(action[0])
            alpha = alpha_function(timestep=timestep)
            V[state] += alpha * (reward + (gamma * V[next_state]) - V[state])
            state = next_state
            timestep += 1
        return V


if __name__ == '__main__':
    env = EnvQ(timestep_limit=10e+4, seed=SEED)
    td = TD(env)
    V = td.evaluate(policy=get_aggressive_policy, alpha_function=alpha_function)
    print(V)
