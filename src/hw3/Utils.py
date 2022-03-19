import os

import seaborn as sns

from EnvQ import ACTION_HIGH, STATE_SIZE
from collections import OrderedDict
from matplotlib import pyplot as plt


def plot_combination(dict_, tag="", type="scatter", default_folder="images", scale='normal'):
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
    fig1 = plt.gcf()
    ax = plt.gca()
    if scale == 'log':
        ax.set_yscale('log')
        # ax.set_xscale('log')  
    plt.show()
    if len(tag) >0:
        if not os.path.exists(default_folder):
            os.makedirs(default_folder)
        fig1.savefig(default_folder+"/"+tag+"_"+type+'.png')

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


def plot_policy(policy, label="", tag="", type="A", default_folder="images"):
    q_high = [row[ACTION_HIGH] for row in policy]
    y_line = list(range(STATE_SIZE))
    if type ==  "A":
        sns.set_theme(style="whitegrid")
        sns.set_palette("Set2")
    else:
        sns.set_theme(style="darkgrid", font='Latin Modern Roman')
        sns.set_palette("husl")   
    plt.scatter(y_line, q_high, alpha=0.9, label=label)    
    plt.legend()
    plt.title(tag)
    fig1 = plt.gcf()
    plt.show()
    if len(tag) >0:
        if not os.path.exists(default_folder):
            os.makedirs(default_folder)
        fig1.savefig(default_folder+"/"+tag+"_"+type+'.png')
    


def plot_difference(v1, v2, tag=""):
    zip_object = zip(v1, v2)
    difference = []
    for v_1, v_2 in zip_object:
        difference.append(v_1 - v_2)
    plt.bar(range(len(difference)), difference)
    plt.title(tag)
    plt.show()

if __name__ == '__main__':
    d = {'help': [-663.3323,  -663.4832, -1150.8296],
         'im not ok': [1.e-02, 1.e+00, 1.e+02]
         }

    plot_combination(d, tag="Ayuda", type="line", scale="log")
    # plot_combination(d, tag="Ayuda")