import random
import math
import numpy as np

from tqdm import trange
from ReplayBuffer import ReplayBuffer
from EnvQ import EnvQ, DISCOUNT_FACTOR
from Utils import plot_x_y
from EnvQ import EnvQ, STATE_SIZE, NUM_ACTION
from LSTD import LSTD
from collections import deque


class SoftPolicyIteration():

    def __init__(self, t=1e5, k=100, seed=42):
        self.seed = seed
        np.random.seed(seed=seed)
        random.seed(seed)
        self.t = t
        self.k = k
        self.replay_buffer = ReplayBuffer()
        self.env = EnvQ(timestep_limit=self.t)
        self.state_size = self.env.max_length
        self.policy = self.get_random_policy()

    def get_random_policy(self):
        policy = np.random.rand(STATE_SIZE, NUM_ACTION)
        policy[:, 1] = 1 - policy[:, 0]
        return policy

    def get_lazy_policy(self):
        policy = np.ones((STATE_SIZE, NUM_ACTION))
        policy[:, 1] = 1 - policy[:, 0]
        return policy

    def collect_transitions(self):
        done = False
        state = self.env.q3_reset()
        self.replay_buffer.clear()
        total_reward = 0
        while not done:
            action = np.random.choice(a=self.env.actions, p=self.policy[state])
            next_state, reward, done, _ = self.env.step(action=action)
            total_reward += reward
            self.replay_buffer.insert(stateV=state, actonV=action, next_stateV=next_state, rewardV=reward, doneV=done)
            state = next_state
        # total_reward = self.replay_buffer.buffer['reward'].sum()
        return total_reward

    def get_q(self, shift_q=True):
        lstd = LSTD(env=self.env, seed=self.seed)
        q = lstd.get_q_estimate(rb=self.replay_buffer.buffer)
        if shift_q:
            for s in range(self.env.max_length):
                actions = q[s]
                max_val = np.max(actions)
                q[s] -= max_val
        return q

    def update_policy(self, eta):
        policy = np.ones((STATE_SIZE, NUM_ACTION))
        q = self.get_q()

        for s in range(self.env.max_length):
            total_sum = 0
            policy[s] = [0, 0]

            for action in range(len(policy[s])):
                exponent = math.exp(eta * q[s][action])
                total_sum += self.policy[s][action] * exponent

            for action in range(len(policy[s])):
                exponent = math.exp(eta * q[s][action])
                total_by_action = self.policy[s][action] * exponent / total_sum
                policy[s][action] = total_by_action

        self.policy = policy

    def iteration(self, eta):
        reward = 0
        dq = deque([], 5)
        for i in trange(self.k):
            r = self.collect_transitions()
            reward += r
            self.get_q()
            self.update_policy(eta=eta)

            # Hack to run faster
            dq.append(np.copy(self.policy))
            if self.all_policy_same(dq=dq):
                reward += (r * (self.k - i))
                break
            # Hack over
        return reward

    def all_policy_same(self, dq: deque):
        capacity = dq.maxlen
        items_count = len(dq)
        if items_count < capacity:
            return False
        difference_threshold = 1e-6  # Hack value
        p = dq[0]
        for ps in dq:
            diff_between_arrays = np.absolute(np.array(p) - np.array(ps))
            diff = np.sum(diff_between_arrays)
            real_difference_flag = diff > difference_threshold
            if real_difference_flag:
                return False
        return True


def q3():
    rewards = []
    m_val = np.logspace(-2, 2, num=5)  # 100
    # m_val = [1e2]

    for m in m_val:
        spi = SoftPolicyIteration(t=1e3, k=100)
        r = spi.iteration(eta=m)
        rewards.append(r)
    print(rewards)
    print(m_val)
    plot_x_y(m_val, rewards, scale='log', tag="Soft Policy Iteration")


if __name__ == '__main__':
    q3()
