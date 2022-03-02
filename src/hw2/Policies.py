import time
import copy
import random
import numpy as np
from matplotlib import pyplot as plt

from EnvQ import EnvQ
from IterativePolicyEvaluation import IterativePolicyEvaluation
from Utils import plot_dict, plot_list

ACTION_LOW = 0
ACTION_HIGH = 1
NUM_ACTION = 2
STATE_SIZE = 100
DISCOUNT_FACTOR = 0.9
CHECKPOINT_MARKS = [10, 20, 50, 100]


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


def policy_improvement(env: EnvQ, V, gamma=DISCOUNT_FACTOR):
    policy = np.zeros([env.max_length, NUM_ACTION]) / NUM_ACTION
    for s in range(env.max_length):
        q = q_from_v(env, V, s, gamma=gamma)
        # OPTION 1: construct a deterministic policy
        # policy[s][np.argmax(q)] = 1

        # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
        best_a = np.argwhere(q == np.max(q)).flatten()
        policy[s] = np.sum([np.eye(NUM_ACTION)[i] for i in best_a], axis=0) / len(best_a)

    return policy


def policy_iteration(env: EnvQ, ipe: IterativePolicyEvaluation, gamma=DISCOUNT_FACTOR, theta=1e-8):
    start_time = time.process_time()
    time_100 = np.inf
    policy = np.ones([env.max_length, NUM_ACTION]) / NUM_ACTION
    value_function_dict = {}
    iteration_count = 0
    # print(policy)
    while True:
        V = ipe.evaluate(policy=policy, gamma=gamma, theta=theta)
        if iteration_count in CHECKPOINT_MARKS:
            value_function_dict[iteration_count] = copy.copy(V)

        new_policy = policy_improvement(env, V)

        # OPTION 1: stop if the policy is unchanged after an improvement step
        # if (new_policy == policy).all():
        #    break;

        # OPTION 2: stop if the value function estimates for successive policies has converged
        if np.max(abs(ipe.evaluate(policy) - ipe.evaluate(new_policy))) < theta:
            break

        policy = copy.copy(new_policy)
        iteration_count += 1
        if iteration_count == 100:
            time_100 = (time.process_time() - start_time)

    # print(policy)
    time_taken_s = (time.process_time() - start_time)
    time_100 = min(time_taken_s, time_100)
    return policy, V, iteration_count, value_function_dict, time_taken_s, time_100


def value_iteration(env: EnvQ, gamma=DISCOUNT_FACTOR, theta=1e-8):
    start_time = time.process_time()
    time_100 = np.inf
    V = np.zeros(env.max_length)
    value_function_dict = {}
    iteration_count = 0
    while True:
        delta = 0
        for s in range(env.max_length):
            v = V[s]
            V[s] = max(q_from_v(env, V, s, gamma))
            delta = max(delta, abs(V[s] - v))
        if delta < theta:
            break
        # print(delta)
        iteration_count += 1
        if iteration_count == 100:
            time_100 = (time.process_time() - start_time)

        if iteration_count in CHECKPOINT_MARKS:
            value_function_dict[iteration_count] = copy.copy(V)

    value_function_dict[iteration_count] = copy.copy(V)
    policy = policy_improvement(env, V, gamma)
    time_taken_s = (time.process_time() - start_time)
    time_100 = min(time_taken_s, time_100)
    return policy, V, iteration_count, value_function_dict, time_taken_s, time_100


def plot_difference(v1, v2, tag=""):
    zip_object = zip(v1, v2)
    difference = []
    for v_1, v_2 in zip_object:
        difference.append(v_1 - v_2)

    plt.bar(range(len(difference)), difference)
    plt.title(tag)
    plt.show()


def plot_policy(policy, label="", tag=""):
    # q_high = policy[ACTION_HIGH]
    q_high = [row[ACTION_HIGH] for row in policy]
    y_line = list(range(STATE_SIZE))
    plt.scatter(y_line, q_high, alpha=0.9, label=label)
    plt.legend()
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
    print("Difference at timestep 49 is ", v_lazy[49] - v_aggressive[49])
    print("Difference at timestep 50 is ", v_lazy[50] - v_aggressive[50])
    print("Difference at timestep 80 is ", v_lazy[80] - v_aggressive[80])

    y_line = list(range(env.max_length))
    plt.scatter(y_line, v_lazy, alpha=0.5, label="Lazy Policy Value Function")
    plt.scatter(y_line, v_aggressive, alpha=0.5, label="Aggressive Policy Value Function")
    plt.title('Lazy and Aggressive Policies')
    plt.legend()
    plt.show()

    plot_policy(policy=lp, label="Action (0 = Low, 1 = High)", tag="Lazy Policy")
    plot_policy(policy=ap, label="Action (0 = Low, 1 = High)", tag="Aggressive Policy")

    return v_lazy, v_aggressive


def problem2(lp_v, ap_v):
    env = EnvQ()
    ipe = IterativePolicyEvaluation(env=env)
    pi_p, pi_v, pi_steps, pi_checkpoints, pi_time, pi_time_100 = policy_iteration(env=env, ipe=ipe)
    vi_p, vi_v, vi_steps, vi_checkpoints, vi_time, vi_time_100 = value_iteration(env=env)

    print("Policy iteration took ", pi_steps, " iterations and ", pi_time, " s")
    print("Value iteration took  ", vi_steps, " iterations and ", vi_time, " s")

    print("Policy Iteration time for 100 steps ", pi_time_100)
    print("Value Iteration time for 100 steps ", vi_time_100)

    ipe.plot_value_function(pi_v, tag="Policy Iteration Value Function")
    ipe.plot_value_function(vi_v, tag="Value Iteration Value Function")

    y_line = list(range(env.max_length))
    pi_line = pi_v
    v_line = vi_v
    plt.scatter(y_line, pi_line, alpha=0.8, label="Policy Iteration: Final")
    plt.scatter(y_line, v_line, alpha=0.8, label="Value Iteration: Final")

    for timestep in CHECKPOINT_MARKS:
        vi_value = vi_checkpoints[timestep] if timestep in vi_checkpoints else vi_v
        tag = "Value Iteration: " + str(timestep)
        plt.scatter(y_line, vi_value, alpha=0.8, label=tag)

    plt.title('PI & VI Value Functions over different timesteps')
    plt.legend()
    plt.show()

    plot_difference(pi_v, vi_v, "Policy Iteration - Value Iteration")
    plot_difference(pi_v, lp_v, "Optimal - Lazy Policy")
    plot_difference(pi_v, ap_v, "Optimal - Aggressive Policy")
    plot_policy(policy=pi_p, label="Action (0 = Low, 1 = High)", tag="Optimal Policy")


if __name__ == '__main__':
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    lp_v, ap_v = problem1()
    problem2(lp_v=lp_v, ap_v=ap_v)
