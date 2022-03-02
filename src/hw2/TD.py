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

    def __init__(self, env: EnvQ,  gamma=DISCOUNT_FACTOR):
        self.env = env
        self.gamma = gamma

    def get_alpha(self, timestep, a=10e5, b=10e5):
        alpha = a / (timestep + b)
        return alpha

    def get_v(self, theta: np.ndarray, feature_map: np.ndarray):
        V = defaultdict(float)
        for state in range(self.env.max_length):
            V[state] = float(np.matmul(theta.T, feature_map[state]))
        return V

    def evaluate(self, policy, actual_value):
        v = defaultdict(float)
        state = self.env.reset()
        done = False
        timestep = 0
        pbar = tqdm(total=(self.env.timestep_limit + 1))
        while not done:
            action = np.random.choice(a=self.env.actions, p=policy[state])
            # action = action[0]
            next_state, reward, done, _ = self.env.step(action=action)
            alpha = self.get_alpha(timestep=timestep)
            delta = (reward + self.gamma * v[next_state]) - v[state]
            v[state] += (alpha * delta)
            state = next_state
            if timestep % 10000 == 0 and len(v) == len(actual_value):
                # update MSE every 1000 steps
                value = list(v.values())
                diff = mean_squared_error(value, actual_value)
                pbar.set_description("MSE Diff is %f" % diff)
            timestep += 1
            pbar.update(1)
        pbar.close()
        return v


if __name__ == '__main__':
    env = EnvQ(timestep_limit=10e+4, seed=SEED)
    td = TD(env)
    fm = FeatureMaps()
    fine_map = fm.get_fine_fm()
    coarse_map = fm.get_coarse_fm()
    pwl_map = fm.get_pwl_fm()

    policy = get_lazy_policy()
    ipe = IterativePolicyEvaluation(env=td.env)
    v_lazy = ipe.evaluate(policy=policy, gamma=DISCOUNT_FACTOR)
    V = td.evaluate(policy=policy, actual_value=v_lazy)
    plot_dict(V,"TD0")

    '''
    V = td.evaluate(policy=get_lazy_policy, feature_map=fine_map, alpha_function=alpha_function)
    print(V)
    plot_dict(a=V, tag="Approximate Value Function")
    '''
