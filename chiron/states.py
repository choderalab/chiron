from openmm import unit
from typing import List, Optional
from jax import numpy as jnp
from loguru import logger as log
from .potential import NeuralNetworkPotential
from openmm.app import Topology


class SamplerState:
    """
    Represents the state that is changed by the integrator.
    Parameters
    ----------
    positions : Nx3 openmm.unit.Quantity
        Position vectors for N particles (length units).
    velocities : Nx3 openmm.unit.Quantity, optional
        Velocity vectors for N particles (velocity units).
    box_vectors : 3x3 openmm.unit.Quantity
        Current box vectors (length units).

    """

    def __init__(self, positions, velocities=None, box_vectors=None) -> None:
        self.positions = positions
        self.velocities = velocities
        self.box_vectors = box_vectors

    @property
    def unitless_positions(self):
        return self.positions.value_in_unit_system(unit.md_unit_system)


class ThermodynamicState:
    """
    Represents the thermodynamic ensemble

    Attributes
    ----------
    potential : NeuralNetworkPotential
        The potential energy function of the system.
    temperature : unit.Quantity
        The temperature of the simulation.
    volume : unit.Quantity
        The volume of the simulation.
    pressure : unit.Quantity
        The pressure of the simulation.

    """

    def __init__(
        self,
        potential: Optional[NeuralNetworkPotential],
        temperature: Optional[unit.Quantity] = None,
        volume: Optional[unit.Quantity] = None,
        pressure: Optional[unit.Quantity] = None,
    ):
        self.potential = potential
        self.temperature = temperature
        self.volume = volume
        self.pressure = pressure

        from .utils import get_nr_of_particles

        self.nr_of_particles = get_nr_of_particles(self.potential.topology)
        self._check_completness()

    def check_variables(self):
        """
        Check which variables in the __init__ method are None.

        Returns
        -------
        List[str]
            A list of variable names that are None.
        """
        variables = [
            "temperature",
            "volume",
            "pressure",
        ]
        set_variables = [var for var in variables if getattr(self, var) is not None]
        return set_variables

    def _check_completness(self):
        # check which variables are set
        set_variables = self.check_variables()

        if len(set_variables) == 0:
            log.info("No variables are set.")

        # print all set variables
        for var in set_variables:
            log.info(f"{var} is set.")

        if self.temperature and self.volume and self.nr_of_particles:
            log.info("NVT ensemble simulated.")
        if self.temperature and self.pressure and self.nr_of_particles:
            log.info("NpT ensemble is simulated.")

    @classmethod
    def are_states_compatible(cls, state1, state2):
        """
        Check if two simulation states are compatible.

        This method should define the criteria for compatibility,
        such as matching number of particles, etc.

        Parameters
        ----------
        state1 : SimulationState
            The first simulation state to compare.
        state2 : SimulationState
            The second simulation state to compare.

        Returns
        -------
        bool
            True if states are compatible, False otherwise.
        """
        pass

    def get_reduced_potential(self, sampler_state: SamplerState):
        """Compute the reduced potential in this thermodynamic state.

        Parameters
        --------
        sampler_state : SamplerState
            The sampler state to compute the reduced potential for.
            Contains positions and box vectors.

        Returns
        -------
        u : float
            The unit-less reduced potential, which can be considered
            to have units of kT.

        Notes
        -----
        The reduced potential is defined as in Ref. [1],

        u = \beta [U(x) + p V(x) + \mu N(x)]

        where the thermodynamic parameters are

        \beta = 1/(kB T) is the inverse temperature
        p is the pressure
        \mu is the chemical potential

        and the configurational properties are

        x the atomic positions
        U(x) is the potential energy
        V(x) is the instantaneous box volume
        N(x) the numbers of various particle species (e.g. protons of
             titratable groups)

        References
        ----------
        [1] Shirts MR and Chodera JD. Statistically optimal analysis of
        equilibrium states. J Chem Phys 129:124105, 2008.

        Examples
        --------
        Compute the reduced potential of a water box at 298 K and 1 atm.

        >>> from openmmtools import testsystems
        >>> waterbox = testsystems.WaterBox(box_edge=20.0*unit.angstroms)
        >>> topology, positions = waterbox.system.topology, waterbox.positions
        >>> state = SimulationsState(temperature=298.0*unit.kelvin,
        ...                            pressure=1.0*unit.atmosphere, position=waterbox.positions)
        >>> u = state.reduced_potential()
        """
        beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * (self.temperature * unit.kelvin))
        reduced_potential = (
            self.potential.compute_energy(sampler_state.unitless_positions)
            * unit.kilocalories_per_mole
        ) / unit.AVOGADRO_CONSTANT_NA

        if self.pressure is not None:
            reduced_potential += self.pressure * self.volume

        return beta * reduced_potential
