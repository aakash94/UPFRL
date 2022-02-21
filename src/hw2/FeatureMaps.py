import time
import copy
import random
import numpy as np

from EnvQ import EnvQ


class FeatureMaps():

    def __init__(self):
        print("Feature Maps")
        env = EnvQ()
        self.len_q = env.max_length

    def get_fine_fm(self):
        shape = (self.len_q+1, self.len_q+1)
        fm = np.zeros(shape)
        np.fill_diagonal(fm, 1)
        return fm

    def get_coarse_fm(self):
        return []

    def get_pwl_fm(self):
        return []


if __name__ == '__main__':
    fm = FeatureMaps()
    ffm = fm.get_fine_fm()
    cfm = fm.get_coarse_fm()
    pwlfm = fm.get_pwl_fm()
    print("\nFine\n", ffm)
    print("\nCoarse\n", cfm)
    print("\npiecewise linear\n", pwlfm)
