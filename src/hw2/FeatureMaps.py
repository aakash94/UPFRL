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
        side = self.len_q+1
        shape = (side, side)
        fm = np.zeros(shape)
        np.fill_diagonal(fm, 1)
        return fm

    def get_coarse_fm(self):
        side = int(self.len_q/5)
        shape = (self.len_q, side)
        fm = np.zeros(shape)

        for x in range(self.len_q+1):
            for s in range(side):
                lower_bound = 5*(s-1)
                upper_bound = (5*s)-1
                if x>= lower_bound and x <= upper_bound:
                    fm[x][s] = 1
        return fm

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
