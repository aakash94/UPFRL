import gym
import random


class EnvQ(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_state=100, min_state=0, action_size=2, timestep_limit=1073741824, seed=None):
        if seed is not None:
            random.seed(seed)

        self.timestep_limit = timestep_limit
        self.max_length = max_state
        self.min_length = min_state
        self.state = random.randint(0, 99)

        self.action_low = 0
        self.action_high = 1

        self.q_action = {self.action_low: 0.51, self.action_high: 0.6}
        self.cost_action = {self.action_low: 0, self.action_high: 0.01}

        self.timestep_count = 0

    def step(self, action: int):
        self.timestep_count += 1
        done = False
        new_state = self.state

        if self.timestep_count == self.timestep_limit:
            done = True

        reward = 0 - (((self.state / self.max_length) ** 2) + (self.cost_action[action]))

        arrival_rate = 0.5
        service_rate = self.q_action[action]

        if random.random() < arrival_rate:
            new_state += 1

        if random.random() < service_rate:
            new_state -= 1

        new_state = max(min(new_state, self.max_length), self.min_length)
        self.state = new_state
        return self.state, reward, done, {}

    def reset(self):
        self.state = random.randint(0, 99)
        self.timestep_count = 0
        return self.state

    def render(self, mode="human"):
        print(self.state)

    def close(self):
        pass


if __name__ == '__main__':
    env = EnvQ()
    env.render()
    env.step(env.action_low)
    env.render()
    env.step(env.action_high)
    env.render()
