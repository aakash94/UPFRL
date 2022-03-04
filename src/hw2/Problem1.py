import numpy as np

from EnvQ import EnvQ
from Policies import get_lazy_policy, get_aggressive_policy, DISCOUNT_FACTOR, plot_policy
from Utils import plot_combination
from FeatureMaps import FeatureMaps
from TD import TD
from LSTD import LSTD
from IterativePolicyEvaluation import IterativePolicyEvaluation
from ApproximatePolicyIteration import ApproximatePolicyIteration

iterations = [1e4, 1e5, 1e6, 1e7]


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
        ep_string = str(int(i))
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

        tag = policy_name + "_fine_" + ep_string
        td_values[tag] = fine_td_v.values()
        tag = policy_name + "_coarse_" + ep_string
        td_values[tag] = coarse_td_v.values()
        tag = policy_name + "_pwl_" + ep_string
        td_values[tag] = pwl_td_v.values()

        tag = policy_name + "_fine_" + ep_string
        lstd_values[tag] = fine_lstd_v.values()
        tag = policy_name + "_coarse_" + ep_string
        lstd_values[tag] = coarse_lstd_v.values()
        tag = policy_name + "_pwl_" + ep_string
        lstd_values[tag] = pwl_lstd_v.values()

    return td_values, lstd_values


def get_plot_name(plot_name="", tag_prefix=""):
    parts = plot_name.split('_')
    policy_name = parts[0]
    fm_name = parts[1]
    final_name = tag_prefix + " Policy : " + policy_name + " Feature Map : " + fm_name
    return final_name


def main():
    # Get the policies
    lazy_policy = get_lazy_policy()
    aggressive_policy = get_aggressive_policy()

    env = EnvQ(timestep_limit=1e5)
    fm = FeatureMaps()
    fine_map = fm.get_fine_fm()
    pi = ApproximatePolicyIteration(env=env, k=10)
    api_policy_10 = pi.policy_iteraion(feature_map=fine_map)
    plot_policy(policy=api_policy_10, label="Action", tag="API Policy 10", type="A")
    plot_policy(policy=api_policy_10, label="Action", tag="API Policy 10", type="B")
    api_v_10 = pi.get_value_function(policy=api_policy_10, feature_map=fine_map)

    pi = ApproximatePolicyIteration(env=env, k=100)
    api_policy_100 = pi.policy_iteraion(feature_map=fine_map)
    plot_policy(policy=api_policy_100, label="Action", tag="API Policy 100", type="A")
    plot_policy(policy=api_policy_100, label="Action", tag="API Policy 100", type="B")
    api_v_100 = pi.get_value_function(policy=api_policy_100, feature_map=fine_map)

    # get value functions from of the policies from 1st lab
    lazy_ipe = get_ipe_v(policy=lazy_policy)
    aggressive_ipe = get_ipe_v(policy=aggressive_policy)

    # get all the new values estimates using td and lstd for lazy policy
    lazy_td_v, lazy_lstd_v = get_values(policy=lazy_policy, policy_name="Lazy")

    # add ipe values to the dict
    lazy_td_v["Lazy_ipe"] = lazy_ipe
    lazy_lstd_v["Lazy_ipe"] = lazy_ipe

    lazy_td_v["api_api_10"] = api_v_10
    lazy_td_v["api_api_100"] = api_v_100

    lazy_lstd_v["api_api_10"] = api_v_10
    lazy_lstd_v["api_api_100"] = api_v_100
    print("Lazy Policy Done")

    td_lazy_fine = {}
    td_lazy_coarse = {}
    td_lazy_pwl = {}

    lstd_lazy_fine = {}
    lstd_lazy_coarse = {}
    lstd_lazy_pwl = {}

    td_aggressive_fine = {}
    td_aggressive_coarse = {}
    td_aggressive_pwl = {}

    lstd_aggressive_fine = {}
    lstd_aggressive_coarse = {}
    lstd_aggressive_pwl = {}

    for name, values in lazy_td_v.items():
        if 'fine' in name:
            td_lazy_fine[name] = values

        elif 'coarse' in name:
            td_lazy_coarse[name] = values

        elif 'pwl' in name:
            td_lazy_pwl[name] = values

        elif 'ipe' in name:
            td_lazy_fine[name] = values
            td_lazy_coarse[name] = values
            td_lazy_pwl[name] = values

        elif 'api' in name:
            td_lazy_fine[name] = values
            td_lazy_coarse[name] = values
            td_lazy_pwl[name] = values

    for name, values in lazy_lstd_v.items():
        if 'fine' in name:
            lstd_lazy_fine[name] = values

        elif 'coarse' in name:
            lstd_lazy_coarse[name] = values

        elif 'pwl' in name:
            lstd_lazy_pwl[name] = values

        elif 'ipe' in name:
            lstd_lazy_fine[name] = values
            lstd_lazy_coarse[name] = values
            lstd_lazy_pwl[name] = values

        elif 'api' in name:
            lstd_lazy_fine[name] = values
            lstd_lazy_coarse[name] = values
            lstd_lazy_pwl[name] = values

    # get all the new values estimates using td and lstd for aggressive policy
    aggressive_td_v, aggressive_lstd_v = get_values(policy=aggressive_policy, policy_name="Aggressive")
    # add ipe policy to the dict
    aggressive_td_v["Aggressive_ipe"] = aggressive_ipe
    aggressive_lstd_v["Aggressive_ipe"] = aggressive_ipe

    aggressive_td_v["api_api_10"] = api_v_10
    aggressive_td_v["api_api_100"] = api_v_100

    aggressive_lstd_v["api_api_10"] = api_v_10
    aggressive_lstd_v["api_api_100"] = api_v_100
    print("Aggressive Policy Done")

    for name, values in aggressive_td_v.items():
        if 'fine' in name:
            td_aggressive_fine[name] = values

        elif 'coarse' in name:
            td_aggressive_coarse[name] = values

        elif 'pwl' in name:
            td_aggressive_pwl[name] = values

        elif 'ipe' in name:
            td_aggressive_fine[name] = values
            td_aggressive_coarse[name] = values
            td_aggressive_pwl[name] = values

        elif 'api' in name:
            td_aggressive_fine[name] = values
            td_aggressive_coarse[name] = values
            td_aggressive_pwl[name] = values

    for name, values in aggressive_lstd_v.items():
        if 'fine' in name:
            lstd_aggressive_fine[name] = values

        elif 'coarse' in name:
            lstd_aggressive_coarse[name] = values

        elif 'pwl' in name:
            lstd_aggressive_pwl[name] = values

        elif 'ipe' in name:
            lstd_aggressive_fine[name] = values
            lstd_aggressive_coarse[name] = values
            lstd_aggressive_pwl[name] = values

        elif 'api' in name:
            lstd_aggressive_fine[name] = values
            lstd_aggressive_coarse[name] = values
            lstd_aggressive_pwl[name] = values

    TD_TAG = "TD(0)"
    LSTD_TAG = "LSTD"

    tag = get_plot_name(list(td_lazy_fine.keys())[0], TD_TAG)
    plot_combination(td_lazy_fine, tag)
    plot_combination(td_lazy_fine, tag, "lines")

    tag = get_plot_name(list(td_lazy_coarse.keys())[0], TD_TAG)
    plot_combination(td_lazy_coarse, tag)
    plot_combination(td_lazy_coarse, tag, "lines")

    tag = get_plot_name(list(td_lazy_pwl.keys())[0], TD_TAG)
    plot_combination(td_lazy_pwl, tag)
    plot_combination(td_lazy_pwl, tag, "lines")

    tag = get_plot_name(list(lstd_lazy_fine.keys())[0], LSTD_TAG)
    plot_combination(lstd_lazy_fine, tag)
    plot_combination(lstd_lazy_fine, tag, "lines")

    tag = get_plot_name(list(lstd_lazy_coarse.keys())[0], LSTD_TAG)
    plot_combination(lstd_lazy_coarse, tag)
    plot_combination(lstd_lazy_coarse, tag, "lines")

    tag = get_plot_name(list(lstd_lazy_pwl.keys())[0], LSTD_TAG)
    plot_combination(lstd_lazy_pwl, tag)
    plot_combination(lstd_lazy_pwl, tag, "lines")

    tag = get_plot_name(list(td_aggressive_fine.keys())[0], TD_TAG)
    plot_combination(td_aggressive_fine, tag)
    plot_combination(td_aggressive_fine, tag, "lines")

    tag = get_plot_name(list(td_aggressive_coarse.keys())[0], TD_TAG)
    plot_combination(td_aggressive_coarse, tag)
    plot_combination(td_aggressive_coarse, tag, "lines")

    tag = get_plot_name(list(td_aggressive_pwl.keys())[0], TD_TAG)
    plot_combination(td_aggressive_pwl, tag)
    plot_combination(td_aggressive_pwl, tag, "lines")

    tag = get_plot_name(list(lstd_aggressive_fine.keys())[0], LSTD_TAG)
    plot_combination(lstd_aggressive_fine, tag)
    plot_combination(lstd_aggressive_fine, tag, "lines")

    tag = get_plot_name(list(lstd_aggressive_coarse.keys())[0], LSTD_TAG)
    plot_combination(lstd_aggressive_coarse, tag)
    plot_combination(lstd_aggressive_coarse, tag, "lines")

    tag = get_plot_name(list(lstd_aggressive_pwl.keys())[0], LSTD_TAG)
    plot_combination(lstd_aggressive_pwl, tag)
    plot_combination(lstd_aggressive_coarse, tag, "lines")


if __name__ == '__main__':
    SEED = 666
    np.random.seed(SEED)
    main()
