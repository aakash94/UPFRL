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

    def get_q(self):
        lstd = LSTD(env=self.env, seed=self.seed)
        q = lstd.get_q_estimate(rb=self.replay_buffer.buffer)
        return q

    def update_policy(self, eta):
        policy = np.ones((STATE_SIZE, NUM_ACTION))
        q = self.get_q()
        exponent_clamp = 1e-10  # Hack
        for s in range(self.env.max_length):
            total_sum = 0
            policy[s] = [0, 0]
            for action in range(len(policy[s])):
                exponen = math.exp(eta * q[s][action])
                if exponen < exponent_clamp: exponen = exponent_clamp
                total_sum += self.policy[s][action] * exponen
            for action in range(len(policy[s])):
                exponen = math.exp(eta * q[s][action])
                if exponen < exponent_clamp: exponen = exponent_clamp
                total_by_action = self.policy[s][action] * exponen / total_sum
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
            dq.append(self.policy)
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
        difference_threshold = 1e-6
        p = dq[0]
        for ps in dq:
            if np.absolute(p - ps).sum() > difference_threshold:
                return False
        return True


def q3():
    rewards = []
    m_val = np.logspace(-2, 2, num=5)  # 100
    # m_val = [1e2]

    for m in m_val:
        spi = SoftPolicyIteration(t=1e3, k=10)
        r = spi.iteration(eta=m)
        rewards.append(r)
    print(rewards)
    print(m_val)
    plot_x_y(m_val, rewards, scale='log', tag="Soft Policy Iteration")


if __name__ == '__main__':
    q3()
