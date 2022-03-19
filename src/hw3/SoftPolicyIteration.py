import random
import numpy as np
from tqdm import trange
from ReplayBuffer import ReplayBuffer
from EnvQ import EnvQ, DISCOUNT_FACTOR
from Utils import plot_combination
from EnvQ import EnvQ, STATE_SIZE, NUM_ACTION
from LSTD import LSTD


class SoftPolicyIteration():

    def __init__(self, t=10e2, k=100, seed=42):
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
        # total_reward = self.replay_buffer.buffer['reward'].sum()
        return total_reward

    def get_q(self):
        lstd = LSTD(env=self.env, seed=self.seed)
        q = lstd.get_q_estimate(rb=self.replay_buffer.buffer)
        return q

    def update_policy(self, m):
        # TODO: UPDATE self.policy as per formula given in the pdf
        ...

    def iteration(self, m):
        reward = 0
        for i in trange(self.k):
            reward += self.collect_transitions()
            self.get_q()
            self.update_policy(m=m)
        return reward


def q3():
    rewards = []
    m_val = np.logspace(-2, 2, num=100)

    for m in m_val:
        spi = SoftPolicyIteration()
        r = spi.iteration(m=m)
        rewards.append(r)

    dictionary = {'rewards': rewards, 'Eta Value': m_val}
    plot_combination(dictionary, scale='log')


if __name__ == '__main__':
    q3()
