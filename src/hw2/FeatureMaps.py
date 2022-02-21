import time
import copy
import random
import numpy as np

from EnvQ import EnvQ


class FeatureMaps():

    def __init__(self):
        print("Feature Maps")

    def get_fine_fm(self):
        return []

    def get_coarse_fm(self):
        return []

    def get_pwl_fm(self):
        return []


if __name__ == '__main__':
    fm = FeatureMaps()
    ffm = fm.get_fine_fm()
    cfm = fm.get_coarse_fm()
    pwlfm = fm.get_pwl_fm()
    print("Fine\t", ffm)
    print("Coarse\t", cfm)
    print("piecewise linear\t", pwlfm)
