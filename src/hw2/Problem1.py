import numpy as np

from collections import defaultdict
from tqdm import tqdm

from EnvQ import EnvQ
from Policies import get_lazy_policy, get_aggressive_policy, policy_improvement, DISCOUNT_FACTOR, plot_policy
from Utils import plot_dict
from FeatureMaps import FeatureMaps
from TD import TD
from LSTD import LSTD
from IterativePolicyEvaluation import IterativePolicyEvaluation

iterations = [10e4, 10e5, 10e6, 10e7]


def get_ipe_v(policy):
    env = EnvQ()
    ipe = IterativePolicyEvaluation(env=env)
    v = ipe.evaluate(policy=policy, gamma=DISCOUNT_FACTOR)
    return v


def get_values(policy, policy_name=""):
    td_values = {}
    lstd_values = {}
    fm = FeatureMaps()
    fine_map = fm.get_fine_fm()
    coarse_map = fm.get_coarse_fm()
    pwl_map = fm.get_pwl_fm()

    for i in iterations:
        print("Batch Size \t", i)
        ep_string = str(i)
        env = EnvQ(timestep_limit=i)
        td = TD(env=env)
        lstd = LSTD(env=env)

        fine_td_v = td.evaluate(policy=policy, feature_map=fine_map)
        print("Fine TD done")
        coarse_td_v = td.evaluate(policy=policy, feature_map=coarse_map)
        print("Coarse TD done")
        pwl_td_v = td.evaluate(policy=policy, feature_map=pwl_map)
        print("PWL TD done")

        fine_lstd_v = lstd.evaluate(policy=policy, feature_map=fine_map)
        print("Fine LSTD done")
        coarse_lstd_v = lstd.evaluate(policy=policy, feature_map=coarse_map)
        print("Coarse LSTD done")
        pwl_lstd_v = lstd.evaluate(policy=policy, feature_map=pwl_map)
        print("PWL LSTD done")

        tag = policy_name + "_Fine_" + ep_string
        td_values[tag] = fine_td_v.values()
        tag = policy_name + "_Coarse_" + ep_string
        td_values[tag] = coarse_td_v.values()
        tag = policy_name + "_PWL_" + ep_string
        td_values[tag] = pwl_td_v.values()

        tag = policy_name + "_Fine_" + ep_string
        lstd_values[tag] = fine_lstd_v.values()
        tag = policy_name + "_Coarse_" + ep_string
        lstd_values[tag] = coarse_lstd_v.values()
        tag = policy_name + "_PWL_" + ep_string
        lstd_values[tag] = pwl_lstd_v.values()

    return td_values, lstd_values


def main():
    lazy_policy = get_lazy_policy()
    aggressive_policy = get_aggressive_policy()

    lazy_ipe = get_ipe_v(policy=lazy_policy)
    aggressive_ipe = get_ipe_v(policy=aggressive_policy)

    lazy_td_v, lazy_lstd_v = get_values(policy=lazy_policy, policy_name="Lazy")
    lazy_td_v["Lazy_IPE"] = lazy_ipe.values()
    lazy_lstd_v["Lazy_IPE"] = lazy_ipe.values()
    print("Lazy Policy Done")

    aggressive_td_v, aggressive_lstd_v = get_values(policy=aggressive_policy, policy_name="Aggressive")
    aggressive_td_v["Aggressive_IPE"] = aggressive_ipe.values()
    aggressive_lstd_v["Aggressive_IPE"] = aggressive_ipe.values()
    print("Aggressive Policy Done")


if __name__ == '__main__':
    SEED = 4
    np.random.seed(SEED)
    main()
