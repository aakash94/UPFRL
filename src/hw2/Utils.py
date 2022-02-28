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
