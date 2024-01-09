import numpy as np


class MBAREstimator:
    def __init__(self, N_u: int) -> None:
        self.mbar_f_k = np.zeros(N_u)

    def initialize(self, u_kn: np.ndarray, N_k: np.ndarray):
        """
        Perform mbar analysis
        """
        from pymbar import MBAR
        from loguru import logger as log

        log.debug(f"{N_k=}")
        mbar = MBAR(u_kn=u_kn, N_k=N_k)
        log.debug(mbar.f_k)
        self.mbar_f_k = mbar.f_k
