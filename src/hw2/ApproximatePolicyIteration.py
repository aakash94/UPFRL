from EnvQ import EnvQ
from LSTD import LSTD
from Utils import plot_policy, plot_dict, plot_list
from FeatureMaps import FeatureMaps
from Policies import \
    get_lazy_policy, \
    get_aggressive_policy, \
    policy_improvement, \
    DISCOUNT_FACTOR, \
    STATE_SIZE, \
    NUM_ACTION

from tqdm import tqdm, trange
import numpy as np


class ApproximatePolicyIteration():

    def __init__(self, env: EnvQ, k=100):
        self.env = env
        self.lstd = LSTD(env=self.env)
        self.k = k

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def get_q_estimate(self, v, gamma=DISCOUNT_FACTOR):
        q = np.zeros((STATE_SIZE, NUM_ACTION))
        arrival_rate = self.env.arrival_rate
        for s in range(self.env.max_length):
            for a in range(NUM_ACTION):
                tuples = self.env.transition[s][a]
                d_prob, d_next_state, d_reward, d_done = tuples[0]
                s_prob, s_next_state, s_reward, s_done = tuples[1]
                i_prob, i_next_state, i_reward, i_done = tuples[2]
                service_rate = self.env.q_action[a]
                # all rewards here are same (in the tuples above)
                q[s][a] = s_reward + \
                          (gamma * (1 - arrival_rate) * (service_rate * v[d_next_state])) + \
                          ((1 - service_rate) * v[s_next_state]) + \
                          (gamma * arrival_rate * service_rate * v[s_next_state]) + \
                          ((1 - service_rate) * v[i_next_state])
        return q

    def get_policy(self, q):
        policy = np.ones((STATE_SIZE, NUM_ACTION))
        for s in range(self.env.max_length):
            policy[s] = self.normalize_data(q[s])
        return policy

    def policy_iteraion(self, feature_map):
        v = np.zeros(self.env.max_length)
        q = self.get_q_estimate(v=v)
        policy = self.get_policy(q=q)
        for i in trange(self.k):
            v = self.lstd.evaluate(policy, feature_map, gamma=DISCOUNT_FACTOR)
            q = self.get_q_estimate(v=v)
            policy = self.get_policy(q=q)
        return policy


if __name__ == '__main__':
    from IterativePolicyEvaluation import IterativePolicyEvaluation

    env = EnvQ(timestep_limit=10e4)
    pi = ApproximatePolicyIteration(env=env, k=10)
    op = get_lazy_policy()
    op = get_aggressive_policy()
    # ipe = IterativePolicyEvaluation(env=env)
    # v_ = ipe.evaluate(policy=op, gamma=DISCOUNT_FACTOR)
    # plot_policy(policy=op, label="Original Policy")
    # q = pi.get_q_estimate(v=v_)
    # p = pi.get_policy(q=q)
    # plot_policy(policy=p, label="Derived Policy")
    fm = FeatureMaps()
    fine_map = fm.get_fine_fm()
    coarse_map = fm.get_coarse_fm()
    pwl_map = fm.get_pwl_fm()
    policy = pi.policy_iteraion(feature_map=fine_map)
    plot_policy(policy=policy)