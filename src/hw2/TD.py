import numpy as np

from collections import defaultdict
from tqdm import tqdm

from EnvQ import EnvQ
from Policies import get_lazy_policy, get_aggressive_policy, policy_improvement, DISCOUNT_FACTOR, plot_policy
from IterativePolicyEvaluation import IterativePolicyEvaluation
from Utils import plot_dict

SEED = 4

np.random.seed(SEED)


def alpha_function(timestep=0, a=10 ^ 5, b=10 ^ 5):
    return a / (timestep + b)


class TD():

    def __init__(self, env: EnvQ):
        self.env = env

    def evaluate(self, policy, alpha_function=alpha_function, gamma=DISCOUNT_FACTOR):
        V = defaultdict(float)

        state = self.env.reset()
        policy_fun = policy()
        done = False
        timestep = 0
        pbar = tqdm(desc="Timesteps Elapsed", total=timestep + 1)
        while not done:
            action = np.random.choice(self.env.actions, 1, p=policy_fun[state])
            next_state, reward, done, _ = self.env.step(action[0])
            alpha = alpha_function(timestep=timestep)
            V[state] += alpha * (reward + (gamma * V[next_state]) - V[state])
            state = next_state
            timestep += 1
            pbar.update(1)
        pbar.close()
        return V


if __name__ == '__main__':
    env = EnvQ(timestep_limit=10e+5, seed=SEED)
    td = TD(env)
    V = td.evaluate(policy=get_aggressive_policy, alpha_function=alpha_function)
    print(V)
    plot_dict(a=V, tag="Approximate Value Function")
