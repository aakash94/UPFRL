
from ReplayBuffer import ReplayBuffer
from EnvQ import EnvQ
from Policies import get_get_random_policy
import random
from tqdm import trange
import numpy as np



class SoftPolicyIteration():

    def __init__(self, t=10e5, k=100, seed=42):
        np.random.seed(seed=seed)
        random.seed(seed)
        self.t = t
        self.k = k
        self.replay_buffer = ReplayBuffer()
        self.env = EnvQ(timestep_limit=self.t)
        self.policy = get_get_random_policy()

    def collect_transitions(self):
        done = False
        state = self.env.q3_reset()
        self.replay_buffer.clear()
        while not done:
            action = np.random.choice(a=self.env.actions, p=self.policy[state])
            next_state, reward, done, _ = self.env.step(action=action)
            self.replay_buffer.insert(stateV=state, actonV=action, next_stateV=next_state, rewardV=reward, doneV=done)

    def get_q(self):
        # TODO: Use Replay Buffer to get the Q value function
        # TODO: Maybe use LSTD and PWL feature map as in the question paper
        ...

    def update_policy(self):
        # TODO: UPDATE policy as per formula given in the pdf
        ...

    def iteration(self):
        # TODO: Call functions and Soft Policy Iteration
        for i in trange(self.k):
            self.collect_transitions()
            self.get_q()
            self.update_policy()
