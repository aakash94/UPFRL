import numpy as np

from collections import defaultdict
from tqdm import tqdm

from EnvQ import EnvQ
from Policies import get_lazy_policy, get_aggressive_policy, policy_improvement, DISCOUNT_FACTOR, plot_policy
from IterativePolicyEvaluation import IterativePolicyEvaluation
from Utils import plot_dict
from FeatureMaps import FeatureMaps

SEED = 4

np.random.seed(SEED)


def alpha_function(timestep=0, a=10 ^ 5, b=10 ^ 5):
    return a / (timestep + b)


class TD():

    def __init__(self, env: EnvQ):
        self.env = env

    def get_v(self, theta: np.ndarray, feature_map: np.ndarray):
        V = defaultdict(float)
        for state in range(self.env.max_length):
            V[state] = float(np.matmul(theta.T, feature_map[state]))
        return V

    def evaluate(self, policy, feature_map, alpha_function=alpha_function, gamma=DISCOUNT_FACTOR):
        state = self.env.reset()
        theta = np.zeros_like(feature_map[state])
        policy_fun = policy()
        done = False
        timestep = 0
        pbar = tqdm(desc="Timesteps Elapsed", total=timestep + 1)

        while not done:
            action = np.random.choice(self.env.actions, 1, p=policy_fun[state])
            next_state, reward, done, _ = self.env.step(action[0])

            v_state = np.matmul(theta.T, feature_map[state])
            v_next_state = np.matmul(theta.T, feature_map[next_state])

            alpha = alpha_function(timestep=timestep)
            delta = (reward + (gamma * v_next_state) - v_state)

            theta += alpha * delta * feature_map[state]
            state = next_state

            timestep += 1
            pbar.update(1)
        pbar.close()
        return self.get_v(theta=theta, feature_map=feature_map)


if __name__ == '__main__':
    env = EnvQ(timestep_limit=10e+4, seed=SEED)
    td = TD(env)
    fm = FeatureMaps()
    fine_map = fm.get_fine_fm()
    coarse_map = fm.get_coarse_fm()
    pwl_map = fm.get_pwl_fm()
    V = td.evaluate(policy=get_aggressive_policy, feature_map=fine_map, alpha_function=alpha_function)
    print(V)
    plot_dict(a=V, tag="Approximate Value Function")
