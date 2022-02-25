import time
import copy
import random
import numpy as np

from matplotlib import pyplot as plt
from collections import defaultdict

from sklearn.metrics import mean_squared_error

from tqdm import tqdm

from EnvQ import EnvQ
from Policies import get_lazy_policy, \
    get_aggressive_policy, \
    get_super_aggressive_policy, \
    policy_improvement, \
    DISCOUNT_FACTOR, \
    plot_policy
from IterativePolicyEvaluation import IterativePolicyEvaluation
from FeatureMaps import test_plot

SEED = 4

np.random.seed(SEED)


class TD():

    def __init__(self, env: EnvQ, gamma=DISCOUNT_FACTOR):
        self.env = env
        self.gamma = gamma

    def get_alpha(self, timestep, a=10e5, b=10e5):
        alpha = a / (timestep + b)
        return alpha

    def evaluate(self, policy, actual_value):
        v = defaultdict(float)
        state = self.env.reset()
        done = False
        timestep = 0
        pbar = tqdm(total=(self.env.timestep_limit + 1))
        while not done:
            action = np.random.choice(a=self.env.actions, p=policy[state])
            # action = action[0]
            next_state, reward, done, _ = self.env.step(action=action)
            alpha = self.get_alpha(timestep=timestep)
            delta = (reward + self.gamma * v[next_state]) - v[state]
            v[state] += (alpha * delta)
            state = next_state
            if timestep % 10000 == 0 and len(v) == len(actual_value):
                # update MSE every 1000 steps
                value = list(v.values())
                diff = mean_squared_error(value, actual_value)
                pbar.set_description("MSE Diff is %f" % diff)
            timestep += 1
            pbar.update(1)
        pbar.close()
        return v


if __name__ == '__main__':
    td = TD(env=EnvQ(timestep_limit=10e7, seed=SEED))
    policy = get_super_aggressive_policy()

    ipe = IterativePolicyEvaluation(env=td.env)
    v_lazy = ipe.evaluate(policy=policy, gamma=DISCOUNT_FACTOR)
    V = td.evaluate(policy=policy, actual_value=v_lazy)
    # print(v_lazy)
    # print(V)
    value_v = V.values()
    test_plot(a=value_v, tag="Approximate Value Function")
    test_plot(a=v_lazy, tag="Actual")

    # cost = 0.01
    # cost = 0.0
    # reward = defaultdict(float)
    # for s in range(100):
    #     reward[s] = 0 - (((s / 100) ** 2) + (cost))
    # reward_v = reward.values()
    # test_plot(a=reward_v, tag="Rewards Function")
