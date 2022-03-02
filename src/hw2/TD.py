import numpy as np

from collections import defaultdict
from tqdm import tqdm

from EnvQ import EnvQ
from Policies import get_lazy_policy, get_aggressive_policy, policy_improvement, DISCOUNT_FACTOR, plot_policy
from IterativePolicyEvaluation import IterativePolicyEvaluation
from Utils import plot_dict
from FeatureMaps import FeatureMaps
from sklearn.metrics import mean_squared_error

SEED = 4

np.random.seed(SEED)


def alpha_function(timestep=0, a=10 ^ 5, b=10 ^ 5):
    return a / (timestep + b)


class TD():

    def __init__(self, env: EnvQ):
        self.env = env

    def get_alpha(self, timestep, a=10e5, b=10e5):
        alpha = a / (timestep + b)
        return alpha

    def get_v(self, theta: np.ndarray, feature_map: np.ndarray):
        V = defaultdict(float)
        for state in range(self.env.max_length):
            V[state] = float(np.matmul(theta.T, feature_map[state]))
        return V

    def evaluate(self, policy, feature_map, gamma=DISCOUNT_FACTOR):
        v = defaultdict(float)
        state = self.env.reset()
        theta = np.zeros_like(feature_map[state])
        done = False
        timestep = 0
        pbar = tqdm(total=(self.env.timestep_limit + 1))
        while not done:
            action = np.random.choice(a=self.env.actions, p=policy[state])
            next_state, reward, done, _ = self.env.step(action=action)

            v_state = np.matmul(theta.T, feature_map[state])
            v_next_state = np.matmul(theta.T, feature_map[next_state])

            alpha = self.get_alpha(timestep=timestep)
            delta = (reward + (gamma * v_next_state) - v_state)
            theta += alpha * delta * feature_map[state]

            state = next_state
            timestep += 1
            pbar.update(1)
        pbar.close()
        return self.get_v(theta=theta, feature_map=feature_map)


if __name__ == '__main__':
    env = EnvQ(timestep_limit=10e+5, seed=SEED)
    td = TD(env)
    fm = FeatureMaps()
    fine_map = fm.get_fine_fm()
    coarse_map = fm.get_coarse_fm()
    pwl_map = fm.get_pwl_fm()

    policy = get_lazy_policy()
    V = td.evaluate(policy=policy, feature_map=coarse_map)
    plot_dict(V, "Approximate Value Function")

    '''
    V = td.evaluate(policy=get_lazy_policy, feature_map=fine_map, alpha_function=alpha_function)
    print(V)
    plot_dict(a=V, tag="Approximate Value Function")
    '''
