import numpy as np
from collections import defaultdict
import pandas as pd
from EnvQ import EnvQ, STATE_SIZE, NUM_ACTION, DISCOUNT_FACTOR
from FeatureMaps import FeatureMaps
import random


class LSTD():

    def __init__(self, env: EnvQ, seed=42):
        np.random.seed(seed=seed)
        random.seed(seed)
        self.env = env
        fmaps = FeatureMaps()
        self.fm = fmaps.get_pwl_fm()

    def get_v(self, theta: np.ndarray, feature_map: np.ndarray):
        V = defaultdict(float)
        for state in range(self.env.max_length):
            V[state] = float(np.matmul(theta.T, feature_map[state]))
        return V

    def evaluate_rb(self, rb: pd.DataFrame, gamma=DISCOUNT_FACTOR):
        fm_size = self.fm.shape[1]
        B_T = np.zeros(fm_size)
        A_B = np.zeros((fm_size, fm_size))
        bias = 1e-9

        for ind in rb.index:
            action = rb['action'][ind]
            state = rb['state'][ind]
            next_state = rb['next_state'][ind]
            reward = rb['reward'][ind]

            A_B += self.fm[state].reshape(fm_size, 1) * (self.fm[state] - gamma * self.fm[next_state])
            B_T += self.fm[state] * reward

        if np.linalg.det(A_B) == 0:
            A_B += bias * np.eye(fm_size)

        theta = np.linalg.solve(A_B, B_T)
        v = self.get_v(theta=theta, feature_map=self.fm)
        return v

    def get_q_estimate(self, rb: pd.DataFrame, gamma=DISCOUNT_FACTOR):
        v = self.evaluate_rb(rb=rb)
        q = np.zeros((STATE_SIZE, NUM_ACTION))
        arrival_rate = self.env.arrival_rate
        for s in range(self.env.max_length):
            for a in range(NUM_ACTION):
                tuples = self.env.transition[s][a]
                d_prob, d_next_state, d_reward, d_done = tuples[0]  # decrement
                s_prob, s_next_state, s_reward, s_done = tuples[1]  # same
                i_prob, i_next_state, i_reward, i_done = tuples[2]  # increment
                service_rate = self.env.q_action[a]
                # all rewards here are same (in the tuples above)
                q[s][a] = s_reward + \
                          gamma * (1 - arrival_rate) * (service_rate * v[d_next_state] + \
                                                        (1 - service_rate) * v[s_next_state]) + \
                          gamma * arrival_rate * (service_rate * v[s_next_state] + \
                                                  (1 - service_rate) * v[i_next_state])
        return q


if __name__ == '__main__':
    # from Policies import get_lazy_policy, get_aggressive_policy, policy_improvement, DISCOUNT_FACTOR, plot_policy
    env = EnvQ(timestep_limit=10e+5, seed=SEED)
    lstd = LSTD(env)
    fm = FeatureMaps()
    # fine_map = fm.get_fine_fm()
    # coarse_map = fm.get_coarse_fm()
    pwl_map = fm.get_pwl_fm()
    # policy = get_lazy_policy()
    # V = lstd.evaluate(policy=policy, feature_map=coarse_map)
    # plot_dict(a=V, tag="Approximate Value Function")
