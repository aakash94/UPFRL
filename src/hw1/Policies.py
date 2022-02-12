from EnvQ import EnvQ
from IterativePolicyEvaluation import IterativePolicyEvaluation
import random
import numpy as np
from matplotlib import pyplot as plt

ACTION_LOW = 0
ACTION_HIGH = 1
NUM_ACTION = 2
STATE_SIZE = 100
DISCOUNT_FACTOR = 0.9


def get_aggressive_policy(threshold=50):
    policy = np.zeros((STATE_SIZE, NUM_ACTION))
    for state in range(STATE_SIZE):
        if state < threshold:
            policy[state, ACTION_LOW] = 1

    policy[:, 1] = 1 - policy[:, 0]
    return policy


def get_lazy_policy():
    policy = np.ones((STATE_SIZE, NUM_ACTION))
    policy[:, 1] = 1 - policy[:, 0]
    return policy


def get_action(policy, state):
    action_probability = policy[state]
    actions = [0, 1]
    sampled_action = random.choices(actions, weights=action_probability, k=1)
    action = sampled_action[0]
    return action


class Policy:

    def __init__(self):
        self.action_low = 0
        self.action_high = 1

    def act(self, state):
        return self.action_low


if __name__ == '__main__':
    lp = get_lazy_policy()
    ap = get_aggressive_policy()

    print(lp, ap)
    env = EnvQ()
    ipe = IterativePolicyEvaluation(env=env)

    v_lazy = ipe.evaluate(policy=lp, gamma=DISCOUNT_FACTOR)
    # ipe.plot_value_function(v_lazy)
    # print(v_lazy)

    v_aggressive = ipe.evaluate(policy=ap, gamma=DISCOUNT_FACTOR)
    #ipe.plot_value_function(v_aggressive)
    # print(v_aggressive)
    # plt.plot(v_lazy, label = "lazy")
    # plt.plot(v_aggressive, label = "aggressive")
    # plt.legend()
    # plt.show()

    zip_object = zip(v_lazy, v_aggressive)
    difference = []
    for v_l, v_a in zip_object:
        difference.append(v_l - v_a)
    ipe.plot_value_function(difference)

