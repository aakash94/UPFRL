import copy

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


def q_from_v(env: EnvQ, V, s: int, gamma=DISCOUNT_FACTOR):
    q = np.zeros(NUM_ACTION)
    for a in range(NUM_ACTION):
        for prob, next_state, reward, done in env.transition[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q


def policy_improvement(env: EnvQ, V, gamma=DISCOUNT_FACTOR):
    policy = np.zeros([env.max_length, NUM_ACTION]) / NUM_ACTION
    for s in range(env.max_length):
        q = q_from_v(env, V, s, gamma=gamma)
        best_a = np.argwhere(q == np.max(q)).flatten()
        policy[s] = np.sum([np.eye(NUM_ACTION)[i] for i in best_a], axis=0) / len(best_a)
    return policy


def policy_iteration(env: EnvQ, ipe: IterativePolicyEvaluation, gamma=DISCOUNT_FACTOR, theta=1e-8):
    policy = np.ones([env.max_length, NUM_ACTION]) / NUM_ACTION
    # print(policy)
    while True:
        V = ipe.evaluate(policy=policy, gamma=gamma, theta=theta)
        new_policy = policy_improvement(env, V)

        # OPTION 1: stop if the policy is unchanged after an improvement step
        if (new_policy == policy).all():
            break;

        # OPTION 2: stop if the value function estimates for successive policies has converged
        # if np.max(abs(policy_evaluation(env, policy) - policy_evaluation(env, new_policy))) < theta*1e2:
        #    break;

        policy = copy.copy(new_policy)
    # print(policy)
    return policy, V


def value_iteration(env: EnvQ, gamma=DISCOUNT_FACTOR, theta=1e-8):
    V = np.zeros(env.max_length)
    while True:
        delta = 0
        for s in range(env.max_length):
            v = V[s]
            V[s] = max(q_from_v(env, V, s, gamma))
            delta = max(delta, abs(V[s] - v))
        if delta < theta:
            break
    policy = policy_improvement(env, V, gamma)
    return policy, V


def plot_difference(v1, v2, tag=""):
    zip_object = zip(v1, v2)
    difference = []
    for v_1, v_2 in zip_object:
        difference.append(v_1 - v_2)

    plt.bar(range(len(difference)), difference)
    plt.title(tag)
    plt.show()


def problem1():
    lp = get_lazy_policy()
    ap = get_aggressive_policy()

    env = EnvQ()
    ipe = IterativePolicyEvaluation(env=env)

    v_lazy = ipe.evaluate(policy=lp, gamma=DISCOUNT_FACTOR)
    # ipe.plot_value_function(v_lazy)
    v_aggressive = ipe.evaluate(policy=ap, gamma=DISCOUNT_FACTOR)
    # ipe.plot_value_function(v_aggressive)
    # plt.plot(v_lazy, label = "lazy")
    # plt.plot(v_aggressive, label = "aggressive")
    # plt.legend()
    # plt.show()

    plot_difference(v1=v_lazy, v2=v_aggressive, tag="Lazy - Aggressive")

    return v_lazy, v_aggressive


def problem2(lp_v, ap_v):
    env = EnvQ()
    ipe = IterativePolicyEvaluation(env=env)
    pi_p, pi_v = policy_iteration(env=env, ipe=ipe)
    vi_p, vi_v = value_iteration(env=env)

    plot_difference(pi_v, vi_v, "Policy Iteration - Value Iteration")
    plot_difference(pi_v, lp_v, "Optimal - Lazy Policy")
    plot_difference(pi_v, ap_v, "Optimal - Aggressive Policy")



if __name__ == '__main__':
    lp_v, ap_v = problem1()
    problem2(lp_v=lp_v, ap_v=ap_v)
