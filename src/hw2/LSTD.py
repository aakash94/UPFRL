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


class LSTD():

    def __init__(self, env: EnvQ,  gamma=DISCOUNT_FACTOR):
        self.env = env
        self.gamma = gamma

    def get_v(self, theta: np.ndarray, feature_map: np.ndarray):
        V = defaultdict(float)
        for state in range(self.env.max_length):
            V[state] = float(np.matmul(theta.T, feature_map[state]))
        return V

    def evaluate(self, policy, feature_map):
        state = self.env.reset()
        fm_size = feature_map.shape[1]
        theta = np.zeros_like(feature_map[state])
        B_T = np.zeros(fm_size)
        A_B = np.zeros((fm_size, fm_size))
        bias = 1e-9

        policy_fun = policy()
        done = False
        timestep = 0
        pbar = tqdm(total=(self.env.timestep_limit + 1))

        while not done:
            action = np.random.choice(self.env.actions, p=policy_fun[state])
            next_state, reward, done, _ = self.env.step(action=action)

            A_B += feature_map[state].reshape(fm_size, 1) * (feature_map[state] -self.gamma * feature_map[next_state])
            B_T += feature_map[state] * reward
            state = next_state
            timestep += 1
            pbar.update(1)

        pbar.close()
        if np.linalg.det(A_B) == 0:
            A_B += bias * np.eye(fm_size)

        theta = np.linalg.solve(A_B, B_T)
        return self.get_v(theta=theta, feature_map=feature_map)


if __name__ == '__main__':
    env = EnvQ(timestep_limit=10e+5, seed=SEED)
    lstd = LSTD(env)
    fm = FeatureMaps()
    fine_map = fm.get_fine_fm()
    coarse_map = fm.get_coarse_fm()
    pwl_map = fm.get_pwl_fm()
    V = lstd.evaluate(policy=get_lazy_policy, feature_map=coarse_map)
    print(V)
    plot_dict(a=V, tag="Approximate Value Function")
