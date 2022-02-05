from EnvQ import EnvQ


class AggressivePolicy:

    def __init__(self, threshold=50):
        self.action_low = 0
        self.action_high = 1
        self.threshold = threshold

    def act(self, state):
        if state < self.threshold:
            return self.action_low
        return self.action_high


if __name__ == '__main__':
    env = EnvQ()
    state = env.reset()
    env.render()
    policy = AggressivePolicy()
    env.step(policy.act(state))
    env.render()
