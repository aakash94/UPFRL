from EnvQ import EnvQ


class AggressivePolicy:

    def __init__(self, threshold=50, state_size = 100):
        self.action_low = 0
        self.action_high = 1
        self.threshold = threshold
        self.state_size = state_size

        ones = [1]*self.state_size
        self.action_low_prob = [1 if x<self.threshold else 0 for x in range(self.state_size)]
        self.action_high_prob = [1 - x for x in self.action_low_prob]

    def get_probs(self, state):
        low_prob = self.action_low_prob[state]
        high_prob = self.action_high_prob[state]
        return low_prob, high_prob


    def act(self, state):
        low_prob, high_prob = self.get_probs()
        if state < self.threshold:
            return self.action_low
        return self.action_high


if __name__ == '__main__':
    env = EnvQ()
    state = env.reset()
    env.render()
    policy = AggressivePolicy()
    # env.step(policy.act(state))
    # env.render()
    print(policy.action_low_prob)
    print(policy.action_high_prob)
