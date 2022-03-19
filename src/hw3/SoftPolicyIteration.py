import random
import math
import numpy as np

from tqdm import trange
from ReplayBuffer import ReplayBuffer
from EnvQ import EnvQ, DISCOUNT_FACTOR
from Utils import plot_combination
from EnvQ import EnvQ, STATE_SIZE, NUM_ACTION
from LSTD import LSTD


class SoftPolicyIteration():

    def __init__(self, t=1e3, k=10, seed=42): # t=1e5, k = 100
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
        for s in range(self.env.max_length):
            total_sum = 0
            policy[s] = [0,0]
            for action in range(len(policy[s])):
                total_sum += self.policy[s][action]*math.exp(eta*q[s][action])
            for action in range(len(policy[s])):
                policy[s][action] = self.policy[s][action]*math.exp(eta*q[s][action])/total_sum
        self.policy = policy

    def iteration(self, eta):
        reward = 0
        for i in trange(self.k):
            reward += self.collect_transitions()
            self.get_q()
            self.update_policy(eta=eta)
        return reward


def q3():
    rewards = []
    m_val = np.logspace(-2, 2, num=3) # 100

    for m in m_val:
        spi = SoftPolicyIteration()
        r = spi.iteration(eta=m)
        rewards.append(r)
    plot_combination({ 'Eta Value': m_val}, scale='log')
    plot_combination({'rewards': rewards})


if __name__ == '__main__':
    q3()
