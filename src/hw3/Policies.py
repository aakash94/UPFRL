import time
import copy
import random
import numpy as np
from matplotlib import pyplot as plt

from EnvQ import EnvQ
from Utils import plot_dict, plot_list, plot_policy, plot_difference

ACTION_LOW = 0
ACTION_HIGH = 1
NUM_ACTION = 2
STATE_SIZE = 100
DISCOUNT_FACTOR = 0.9


def get_get_random_policy():
    policy = np.random.rand((STATE_SIZE, NUM_ACTION))
    policy[:, 1] = 1 - policy[:, 0]
    return policy

def get_aggressive_policy(threshold=50):
    policy = np.zeros((STATE_SIZE, NUM_ACTION))
    for state in range(STATE_SIZE):
        if state < threshold:
            policy[state, ACTION_LOW] = 1

    policy[:, 1] = 1 - policy[:, 0]
    return policy
    policy = np.ones((STATE_SIZE, NUM_ACTION))
    policy[:, 1] = 1 - policy[:, 0]
    return policy

def get_lazy_policy():
    policy = np.ones((STATE_SIZE, NUM_ACTION))
    policy[:, 1] = 1 - policy[:, 0]
    return policy


def get_super_aggressive_policy():
    policy = np.ones((STATE_SIZE, NUM_ACTION))
    policy[:, 0] = 1 - policy[:, 1]
    return policy


def get_action(policy, state):
    action_probability = policy[state]
    actions = [0, 1]
    sampled_action = random.choices(actions, weights=action_probability, k=1)
    action = sampled_action[0]
    return action


def q_from_v(env: EnvQ, V, s: int, gamma=DISCOUNT_FACTOR):
    q = np.zeros(NUM_ACTION)
    for a in range(NUM_ACTION):
        for prob, next_state, reward, done in env.transition[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q


if __name__ == '__main__':
    random_seed = 42
