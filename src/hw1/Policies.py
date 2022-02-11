from EnvQ import EnvQ
import random
import numpy as np

ACTION_LOW = 0
ACTION_HIGH = 1
NUM_ACTION = 2
STATE_SIZE = 100


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
    state = 10
    action = ACTION_LOW
    # print(ap)
    # print(ap[state, action])
    # print(ap[50, action])
    get_action(pi=ap, state=60)
