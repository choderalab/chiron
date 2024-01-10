import numpy as np


class MBAREstimator:
    def __init__(self, N_u: int) -> None:
        self.mbar_f_k = np.zeros(N_u)
        self.mbar = None

    def initialize(self, u_kn: np.ndarray, N_k: np.ndarray):
        """
        Perform mbar analysis
        """
        from pymbar import MBAR
        from loguru import logger as log

        log.debug(f"{N_k=}")
        self.mbar = MBAR(u_kn=u_kn, N_k=N_k)

    @property
    def f_k(self):
        from loguru import logger as log

        log.debug(self.mbar.f_k)
        return self.mbar.f_k

    def get_free_energy_difference(self):
        from loguru import logger as log

        log.debug(self.mbar.f_k[-1])
        self.f_k = self.mbar.f_k
        return self.mbar_f_k[-1]
