from EnvQ import ACTION_HIGH, STATE_SIZE
from collections import OrderedDict
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl


def plot_combination(dict_, tag="", type="scatter"):
    # TODO: Plot both versions
    if type == "scatter":
        sns.set_theme(style="whitegrid")
        sns.set_palette("Set2")
    else:
        sns.set_theme(style="darkgrid", font='Latin Modern Roman')
        sns.set_palette("husl")
    for key in dict_:
        y = list(range(len(dict_[key])))
        if type == "scatter":
            plt.scatter(y, dict_[key], label=key)
        else:
            plt.plot(y, dict_[key], label=key)
    plt.tight_layout()
    if type == "scatter":
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=8, rotation=45)      
    plt.legend(prop={'size': 8})
    plt.title(tag, fontweight="bold")
    plt.show()
    #TODO: Save Plot.

def plot_q(q, tag=""):
    action_highs = q[:, 1]
    action_lows = q[:, 0]
    y = list(range(len(action_highs)))
    # plot lines
    plt.plot(y, action_highs, label="High Action")
    plt.plot(y, action_lows, label="Low Action")
    plt.legend()
    plt.title(tag)
    plt.show()


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
    # TODO: Add different Plot
    # TODO: Plot both versions
    q_high = [row[ACTION_HIGH] for row in policy]
    y_line = list(range(STATE_SIZE))
    plt.scatter(y_line, q_high, alpha=0.9, label=label)
    plt.legend()
    plt.title(tag)
    plt.show()
    # TODO: Save Plot


def plot_difference(v1, v2, tag=""):
    zip_object = zip(v1, v2)
    difference = []
    for v_1, v_2 in zip_object:
        difference.append(v_1 - v_2)
    plt.bar(range(len(difference)), difference)
    plt.title(tag)
    plt.show()

if __name__ == '__main__':
    d = {'help': [1, 2, 3, 4 ,6 ,7 ],
         'im not ok': [5, 4 ,3, 10, 12, 21]}

    plot_combination(d, tag="Ayuda", type="line")
    plot_combination(d, tag="Ayuda")