import time
import copy
import random
import numpy as np

from matplotlib import pyplot as plt
from collections import defaultdict

from tqdm import tqdm

from EnvQ import EnvQ
from Policies import get_lazy_policy, get_aggressive_policy, policy_improvement, DISCOUNT_FACTOR, plot_policy
from IterativePolicyEvaluation import IterativePolicyEvaluation
from FeatureMaps import test_plot

SEED = 4

np.random.seed(SEED)


def alpha_function(timestep=0, a=10 ^ 5, b=10 ^ 5):
    return a / (timestep + b)


class TD():

    def __init__(self, env: EnvQ):
        self.env = env

    def evaluate(self, policy, alpha_function=alpha_function, gamma=DISCOUNT_FACTOR):
        V = defaultdict(float)
        coverage = defaultdict(int)
        one_tenth = int(env.timestep_limit/10)
        state = self.env.reset()
        policy_fun = policy()
        done = False
        timestep = 0
        pbar = tqdm(desc="Timesteps Elapsed", total=timestep + 1)
        while not done:
            coverage[state]+=1
            action = np.random.choice(self.env.actions, 1, p=policy_fun[state])
            next_state, reward, done, _ = self.env.step(action[0])
            #alpha = alpha_function(timestep=timestep)
            alpha = 0.001
            delta = (reward + (gamma * V[next_state])) - V[state]
            V[state] += alpha * delta
            state = next_state
            if timestep%one_tenth == 0:
                print("Delta\t", delta)
            timestep += 1
            pbar.update(1)
        pbar.close()
        return V, coverage


if __name__ == '__main__':
    env = EnvQ(timestep_limit=10e5, seed=SEED)
    '''x    
    td = TD(env)
    V, coverage = td.evaluate(policy=get_lazy_policy, alpha_function=alpha_function)
    print(V)
    value_v = V.values()
    coverage_v = coverage.values()
    test_plot(a=coverage_v, tag="Occupancy")
    test_plot(a=value_v, tag="Approximate Value Function")
    '''

    cost = 0.01
    cost = 0.0
    reward = defaultdict(float)
    for s in range(100):
        reward[s] = 0 - (((s / 100) ** 2) + (cost))
    reward_v = reward.values()
    test_plot(a=reward_v, tag="Rewards Function")