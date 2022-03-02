from EnvQ import ACTION_HIGH, STATE_SIZE
from collections import OrderedDict
from matplotlib import pyplot as plt
import seaborn as sns


def plot_dict(a, tag=""):
    sns.set_theme()
    od = OrderedDict(sorted(a.items()))
    plt.bar(range(len(od)), od.values())
    plt.title(tag)
    plt.show()


def plot_list(a, tag=""):
    plt.bar(range(len(a)), a)
    plt.title(tag)
    plt.show()


def plot_policy(policy, label="", tag=""):
    q_high = [row[ACTION_HIGH] for row in policy]
    y_line = list(range(STATE_SIZE))
    plt.scatter(y_line, q_high, alpha=0.9, label=label)
    plt.legend()
    plt.title(tag)
    plt.show()


def plot_difference(v1, v2, tag=""):
    zip_object = zip(v1, v2)
    difference = []
    for v_1, v_2 in zip_object:
        difference.append(v_1 - v_2)
    plt.bar(range(len(difference)), difference)
    plt.title(tag)
    plt.show()
