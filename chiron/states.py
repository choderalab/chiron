from openmm import unit
from typing import List, Optional, Union
from jax import numpy as jnp
from .potential import NeuralNetworkPotential


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
        # NOTE: all units are internally in the openMM units system as documented here:
        # http://docs.openmm.org/latest/userguide/theory/01_introduction.html#units
        if not isinstance(x0, unit.Quantity):
            raise TypeError(f"x0 must be a unit.Quantity, got {type(x0)} instead.")
        if velocities is not None and not isinstance(velocities, unit.Quantity):
            raise TypeError(
                f"velocities must be a unit.Quantity, got {type(velocities)} instead."
            )
        if box_vectors is not None and not isinstance(box_vectors, unit.Quantity):
            if isinstance(box_vectors, List):
                try:
                    box_vectors = self._convert_from_openmm_box(box_vectors)
                except:
                    raise TypeError(f"Unable to parse box_vectors {box_vectors}.")
            else:
                raise TypeError(
                    f"box_vectors must be a unit.Quantity or openMM box, got {type(box_vectors)} instead."
                )
        if not x0.unit.is_compatible(unit.nanometer):
            raise ValueError(f"x0 must have units of distance, got {x0.unit} instead.")
        if velocities is not None and not velocities.unit.is_compatible(
            unit.nanometer / unit.picosecond
        ):
            raise ValueError(
                f"velocities must have units of distance/time, got {velocities.unit} instead."
            )
        if box_vectors is not None and not box_vectors.unit.is_compatible(
            unit.nanometer
        ):
            raise ValueError(
                f"box_vectors must have units of distance, got {box_vectors.unit} instead."
            )
        if box_vectors is not None and box_vectors.shape != (3, 3):
            raise ValueError(
                f"box_vectors must be a 3x3 array, got {box_vectors.shape} instead."
            )

        self._x0 = x0
        self._velocities = velocities
        self._box_vectors = box_vectors
        self._distance_unit = unit.nanometer

    @property
    def n_particles(self) -> int:
        return self._x0.shape[0]

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
    def x0(self, x0: Union[jnp.array, unit.Quantity]) -> None:
        if isinstance(x0, unit.Quantity):
            self._x0 = x0
        else:
            self._x0 = unit.Quantity(x0, self._distance_unit)

    @property
    def distance_unit(self) -> unit.Unit:
        return self._distance_unit

    def _convert_to_jnp(self, array: unit.Quantity) -> jnp.array:
        """
        Convert the sampler state to jnp arrays.
        """
        import jax.numpy as jnp

        array_ = array.value_in_unit_system(unit.md_unit_system)
        return jnp.array(array_)

    def _convert_from_openmm_box(self, openmm_box_vectors: List) -> unit.Quantity:
        box_vec = []
        for i in range(0, 3):
            layer = []
            for j in range(0, 3):
                layer.append(
                    openmm_box_vectors[i][j].value_in_unit(openmm_box_vectors[0].unit)
                )
            box_vec.append(layer)
        return unit.Quantity(jnp.array(box_vec), openmm_box_vectors[0].unit)


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

        if temperature is not None and not isinstance(temperature, unit.Quantity):
            raise TypeError(
                f"temperature must be a unit.Quantity, got {type(temperature)} instead."
            )
        elif temperature is not None:
            if not temperature.unit.is_compatible(unit.kelvin):
                raise ValueError(
                    f"temperature must have units of temperature, got {temperature.unit} instead."
                )

        if volume is not None and not isinstance(volume, unit.Quantity):
            raise TypeError(
                f"volume must be a unit.Quantity, got {type(volume)} instead."
            )
        elif volume is not None:
            if not volume.unit.is_compatible(unit.nanometer**3):
                raise ValueError(
                    f"volume must have units of distance**3, got {volume.unit} instead."
                )
        if pressure is not None and not isinstance(pressure, unit.Quantity):
            raise TypeError(
                f"pressure must be a unit.Quantity, got {type(pressure)} instead."
            )
        elif pressure is not None:
            if not pressure.unit.is_compatible(unit.atmosphere):
                raise ValueError(
                    f"pressure must have units of pressure, got {pressure.unit} instead."
                )

        self.temperature = temperature
        if temperature is not None:
            self.beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * (self.temperature))
        else:
            self.beta = None

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
        from loguru import logger as log

        if len(set_variables) == 0:
            log.info("No variables are set.")

        # print all set variables
        for var in set_variables:
            log.info(f"{var} is set.")

        if self.temperature and self.volume and self.nr_of_particles:
            log.info("NVT ensemble simulated.")
        if self.temperature and self.pressure and self.nr_of_particles:
            log.info("NpT ensemble is simulated.")

    def get_reduced_potential(
        self, sampler_state: SamplerState, nbr_list=None
    ) -> float:
        """
        Compute the reduced potential for the given sampler state.

        Parameters
        ----------
        sampler_state : SamplerState
            The sampler state for which to compute the reduced potential.
        nbr_list : NeighborList or PairList, optional
            The neighbor list or pair list routine to use for calculating the reduced potential.

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
        if self.beta is None:
            self.beta = 1.0 / (
                unit.BOLTZMANN_CONSTANT_kB * (self.temperature * unit.kelvin)
            )
        # log.debug(f"sample state: {sampler_state.x0}")
        reduced_potential = (
            unit.Quantity(
                self.potential.compute_energy(sampler_state.x0, nbr_list),
                unit.kilojoule_per_mole,
            )
        ) / unit.AVOGADRO_CONSTANT_NA
        # log.debug(f"reduced potential: {reduced_potential}")
        if self.pressure is not None:
            reduced_potential += self.pressure * self.volume

        return self.beta * reduced_potential

    def kT_to_kJ_per_mol(self, energy):
        energy = energy * unit.AVOGADRO_CONSTANT_NA
        return energy / self.beta


def calculate_reduced_potential_at_states(
    sampler_state: SamplerState,
    thermodynamic_states: List[ThermodynamicState],
    nbr_list=None,
):
    """
    Calculate the reduced potential for a list of thermodynamic states.

    Parameters
    ----------
    sampler_state : SamplerState
        The sampler state for which to compute the reduced potential.
    thermodynamic_states : list of ThermodynamicState
        The thermodynamic states for which to compute the reduced potential.
    nbr_list : NeighborList or PairList, optional
    Returns
    -------
    list of float
        The reduced potential of the system for each thermodynamic state.

    """
    import numpy as np
    from loguru import logger as log

    reduced_potentials = np.zeros(len(thermodynamic_states))
    for state_idx, state in enumerate(thermodynamic_states):
        reduced_potentials[state_idx] = state.get_reduced_potential(sampler_state)
    log.debug(f"reduced potentials per sampler sate: {reduced_potentials}")
    return reduced_potentials
