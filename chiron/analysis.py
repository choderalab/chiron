import numpy as np


class MBAREstimator:
    def __init__(self) -> None:
        """
        Initialize the MBAR analysis class.

        Returns:
        - None
        """
        self.mbar_f_k = None
        self.mbar = None

    def initialize(self, u_kn: np.ndarray, N_k: np.ndarray):
        """
        Initialize the analysis object.

        Parameters
        ----------
        u_kn: np.ndarray
            Array of dimensionless reduced potentials for each state.
        N_k: np.ndarray
            Array of number of samples for each state.

        """
        from pymbar import MBAR
        from loguru import logger as log

        log.debug(f"{N_k=}")
        self.mbar = MBAR(u_kn=u_kn, N_k=N_k)

    @property
    def f_k(self):
        """
        Free energy for each state.

        Returns
        -------
        mbar.f_k.
        """

        from loguru import logger as log

        log.debug(self.mbar.f_k)
        return self.mbar.f_k

    def get_free_energy_difference(self):
        """
        Calculate the free energy difference between the endstates.

        Returns
        -------
        float
        """
        from loguru import logger as log

        log.debug(self.mbar.f_k[-1])
        self.f_k = self.mbar.f_k
        return self.mbar_f_k[-1]
