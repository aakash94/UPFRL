import random
import numpy as np

from tqdm import trange

from ReplayBuffer import ReplayBuffer
from EnvQ import EnvQ
from Utils import plot_combination
from EnvQ import EnvQ, STATE_SIZE, NUM_ACTION



class SoftPolicyIteration():

    def __init__(self, t=10e2, k=100, seed=42):
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
        while not done:
            action = np.random.choice(a=self.env.actions, p=self.policy[state])
            next_state, reward, done, _ = self.env.step(action=action)
            self.replay_buffer.insert(stateV=state, actonV=action, next_stateV=next_state, rewardV=reward, doneV=done)

        # TODO: Return sum of all rewards
        return 0

    def get_q(self):
        # TODO: Use Replay Buffer to get the Q value function
        # TODO: Maybe use LSTD and PWL feature map as in the question paper
        ...

    def update_policy(self, m):
        # TODO: UPDATE policy as per formula given in the pdf
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
