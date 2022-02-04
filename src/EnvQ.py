import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class AllOnes(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, state_size=4, action_size=2, timestep_limit=1073741824):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
