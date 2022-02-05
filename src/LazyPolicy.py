from EnvQ import EnvQ


class LazyPolicy:

    def __init__(self):
        self.action_low = 0
        self.action_high = 1

    def act(self, state):
        return self.action_low


if __name__ == '__main__':
    env = EnvQ()
    state = env.render()
    policy = LazyPolicy()
    env.step(policy.act(state))
    env.render()
