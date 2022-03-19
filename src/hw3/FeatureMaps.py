import numpy as np

from EnvQ import EnvQ
from Utils import plot_list


class FeatureMaps():

    def __init__(self):
        env = EnvQ()
        self.len_q = env.max_length

    def get_fine_fm(self):
        side = self.len_q
        # side = self.len_q + 1
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
                pos = int(x / 5)
                fm[x][pos] = 1
        return fm

    def get_pwl_fm(self):
        half_side = int(self.len_q / 5)
        side = half_side * 2
        shape = (self.len_q, side)
        cfm = self.get_coarse_fm()
        fm = np.zeros(shape)
        fm[0:100, 0:half_side] = cfm
        for x in range(self.len_q):
            i = int(x / 5)
            pos = half_side + i
            val = (x / 5) - i
            fm[x][pos] = val
        return fm


if __name__ == '__main__':
    fm = FeatureMaps()
    ffm = fm.get_fine_fm()
    cfm = fm.get_coarse_fm()
    pwlfm = fm.get_pwl_fm()

    plot_list(ffm.argmax(axis=1), "FFM")
    plot_list(cfm.argmax(axis=1), "CFM")
    plot_list(pwlfm.argmax(axis=1), "PWLFM First Half")
    plot_list(pwlfm.argmax(axis=1), "PWLFM Second Half")
    print("\nDone\n")
