import numpy as np
import gym
import random
from gym import error, spaces, utils
from gym.utils import seeding


class EnvQ(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_state=100, action_size=2, timestep_limit=1073741824):
        self.timestep_limit = timestep_limit
        self.max_length = max_state
        self.state = random.randint(0, 99)

        self.action_low = 0
        self.action_high = 1

        self.q_action = {self.action_low: 0.51, self.action_high: 0.6}
        self.cost_action = {self.action_low: 0, self.action_high: 0.01}

        self.timestep_count = 0

    def step(self, action):
        reward = 0
        self.timestep_count += 1
        done = False
        if self.timestep_count == self.timestep_limit:
            done = True

        reward = 0 - (((self.state / self.max_length) ** 2) + (self.cost_action[action]))

        return self.state, reward, self.done, {}

    def reset(self):
        self.state = random.randint(0, 99)
        self.timestep_count = 0

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == '__main__':
    env = EnvQ()
