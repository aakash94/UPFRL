import gym
import random


class EnvQ(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_state=100, min_state=0, action_size=2, timestep_limit=10^10, seed=None):
        if seed is not None:
            random.seed(seed)

        self.timestep_limit = timestep_limit
        self.max_length = max_state
        self.min_length = min_state
        self.state = self.max_length - 1

        self.actions = [0, 1]
        self.action_low = 0
        self.action_high = 1

        self.q_action = {self.action_low: 0.51, self.action_high: 0.6}
        self.cost_action = {self.action_low: 0, self.action_high: 0.01}

        self.timestep_count = 0

        self.transition = self.get_env_dynamic()

    def get_env_dynamic(self):
        p = {}
        for s in range(self.min_length, self.max_length):
            p[s] = {}
            for a in self.actions:
                reward = 0 - (((s / self.max_length) ** 2) + (self.cost_action[a]))

                arrival_rate = 0.5
                service_rate = self.q_action[a]

                decrement_p = service_rate * (1 - arrival_rate)
                increment_p = arrival_rate * (1 - service_rate)
                same_p = (service_rate * arrival_rate) + ((1 - service_rate) * (1 - arrival_rate))

                if s >= 99:
                    # state is out of bound, bring back to 99
                    increment_tuple = (increment_p, 99, reward, False)
                else:
                    # state is less than 99, can increment by 1
                    increment_tuple = (increment_p, s + 1, reward, False)

                if s <= 1:
                    # after deduction state will either be 0 or less
                    # bring back to 0 and terminate episode
                    decrement_tuple = (decrement_p, 0, reward, False)
                else:
                    # after deduction state will be greater than 0
                    decrement_tuple = (decrement_p, s - 1, reward, False)

                if s <= 0:
                    # state is already 0 or lesser. Terminate.
                    same_tuple = (same_p, 0, reward, False)
                elif s >= 99:
                    # state has crossed the limit. Bring back in bounds.
                    same_tuple = (same_p, 99, reward, False)
                else:
                    # stay wherever the state is.
                    same_tuple = (same_p, s, reward, False)

                p[s][a] = [decrement_tuple, same_tuple, increment_tuple]

        return p

    def step(self, action: int):
        self.timestep_count += 1
        done = False
        new_state = self.state

        tuples = self.transition[self.state][action]
        d_prob, d_next_state, d_reward, d_done = tuples[0]
        s_prob, s_next_state, s_reward, s_done = tuples[1]
        i_prob, i_next_state, i_reward, i_done = tuples[2]

        random_val = random.random()

        if random_val < d_prob:
            reward = d_reward
            new_state = d_next_state
            done = d_done
        elif random_val >= d_prob and random_val < d_prob + s_prob:
            reward = s_reward
            new_state = s_next_state
            done = s_done
        else:
            reward = i_reward
            new_state = i_next_state
            done = i_done

        if self.timestep_count == self.timestep_limit:
            done = True

        self.state = new_state
        return self.state, reward, done, {}

    def reset(self):
        self.state = self.max_length - 1
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

    # print()
    # print(env.transition)
