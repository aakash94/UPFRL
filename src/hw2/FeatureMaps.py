import time
import copy
import random
import numpy as np
from matplotlib import pyplot as plt

from EnvQ import EnvQ


def test_plot(a, tag = ""):
    plt.bar(range(len(a)), a)
    plt.title(tag)
    plt.show()

class FeatureMaps():

    def __init__(self):
        print("Feature Maps")
        env = EnvQ()
        self.len_q = env.max_length

    def get_fine_fm(self):
        # side = self.len_q
        side = self.len_q + 1
        shape = (side, side)
        fm = np.zeros(shape)
        np.fill_diagonal(fm, 1)
        return fm

    def get_coarse_fm(self):
        side = int(self.len_q / 5)
        shape = (self.len_q, side)
        fm = np.zeros(shape)

        for x in range(self.len_q):
            for s in range(side):
                val = int(x/5)
                fm[x][val] = 1
        return fm

    def get_pwl_fm(self):
        half_side = int(self.len_q / 5)
        side =  half_side*2
        shape = (self.len_q, side)
        cfm = self.get_coarse_fm()
        fm = np.zeros(shape)
        fm[0:100, 0:half_side] = cfm
        return fm


if __name__ == '__main__':
    fm = FeatureMaps()
    ffm = fm.get_fine_fm()
    cfm = fm.get_coarse_fm()
    pwlfm = fm.get_pwl_fm()
    print("\nFine\n", ffm)
    print("\nCoarse\n", cfm)
    print("\npiecewise linear\n", pwlfm)
    print("\nDone\n")
    test_plot(ffm.argmax(axis=1), "FFM")
    test_plot(cfm.argmax(axis=1), "CFM")
    test_plot(pwlfm.argmax(axis=1), "PWLFM")
