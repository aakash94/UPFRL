from EnvQ import EnvQ
from LSTD import LSTD


class ApproximatePolicyIteration():

    def __init__(self, env:EnvQ):
        self.env = EnvQ()
        self.lstd = LSTD(env=self.env)
