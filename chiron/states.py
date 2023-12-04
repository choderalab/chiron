from openmm import unit
from typing import List, Optional
from jax import numpy as jnp
from loguru import logger as log
from .potential import NeuralNetworkPotential
from openmm.app import Topology


class SamplerState:
    """
    Represents the state of the system that is updated during integration.

    Parameters
    ----------
    x0 : unit.Quantity
        The current positions of the particles in the simulation.
    velocities : unit.Quantity, optional
        The velocities of the particles in the simulation.
    box_vectors : unit.Quantity, optional
        The box vectors defining the simulation's periodic boundary conditions.

    """

    def __init__(
        self,
        x0: unit.Quantity,
        velocities: Optional[unit.Quantity] = None,
        box_vectors: Optional[unit.Quantity] = None,
    ) -> None:
        import jax.numpy as jnp

        self._distance_unit = x0.unit
        self._x0 = x0
        self._velocities = velocities
        self._box_vectors = box_vectors

    @property
    def x0(self) -> jnp.array:
        return self._convert_to_jnp(self._x0)

    @property
    def velocities(self) -> jnp.array:
        if self._velocities is None:
            return None
        return self._convert_to_jnp(self._velocities)

    @property
    def box_vectors(self) -> jnp.array:
        if self._box_vectors is None:
            return None
        return self._convert_to_jnp(self._box_vectors)

    @x0.setter
    def x0(self, x0: jnp.array) -> None:
        self._x0 = unit.Quantity(x0, self._distance_unit)

    @property
    def distance_unit(self) -> unit.Unit:
        return self._distance_unit

    def _convert_to_jnp(self, array: unit.Quantity) -> unit.Quantity:
        """
        Convert the sampler state to jnp arrays.
        """
        import jax.numpy as jnp

        array_ = array / self.distance_unit
        return unit.Quantity(jnp.array(array_), self.distance_unit)

    @property
    def x0_unitless(self) -> jnp.ndarray:
        """Return unitless positions."""
        return self.x0.value_in_unit_system(unit.md_unit_system)


class ThermodynamicState:
    """
    Represents the thermodynamic state of the system.

    Parameters
    ----------
    potential : NeuralNetworkPotential
        The potential energy function of the system.
    temperature : unit.Quantity, optional
        The temperature of the simulation.
    volume : unit.Quantity, optional
        The volume of the simulation.
    pressure : unit.Quantity, optional
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

    def check_variables(self) -> None:
        """
        Check if all necessary variables are set and log the simulation ensemble.
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

    def get_reduced_potential(self, sampler_state: SamplerState) -> float:
        """
        Compute the reduced potential for the given sampler state.

        Parameters
        ----------
        sampler_state : SamplerState
            The sampler state for which to compute the reduced potential.

        Returns
        -------
        float
            The reduced potential of the system.

        Notes
        -----
        The reduced potential is computed as:
        u = \beta [U(x) + p V(x) + \mu N(x)],
        where \beta is the inverse temperature, p is the pressure,
        \mu is the chemical potential, x are the atomic positions,
        U(x) is the potential energy, V(x) is the box volume,
        and N(x) is the number of particles.
        """
        beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * (self.temperature * unit.kelvin))
        reduced_potential = (
            self.potential.compute_energy(sampler_state.x0_unitless)
            * unit.kilocalories_per_mole
        ) / unit.AVOGADRO_CONSTANT_NA

        if self.pressure is not None:
            reduced_potential += self.pressure * self.volume

        return beta * reduced_potential
