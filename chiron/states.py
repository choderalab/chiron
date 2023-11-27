from openmm import unit
from typing import List, Optional
from .potential import NeuralNetworkPotential
from jax import numpy as jnp
from loguru import logger as log


class SimulationState:
    """
    Represents the state of a simulation.

    Attributes
    ----------
    temperature : unit.Quantity
        The temperature of the simulation.
    volume : unit.Quantity
        The volume of the simulation.
    pressure : unit.Quantity
        The pressure of the simulation.
    nr_of_particles : int
        The number of particles in the simulation.
    position : unit.Quantity
        The position of the particles in the simulation.

    Methods
    -------
    are_states_compatible(state1, state2)
        Check if two states are compatible and we can compute the reduced potential.
    get_reduced_potential(state)
        Compute the reduced potential for a given state.
    reduced_potential(context_state)
        Compute the reduced potential in this thermodynamic state.

    Examples
    -------
    >>> from openmmtools import testsystems
    >>> waterbox = testsystems.WaterBox(box_edge=20.0*unit.angstroms)
    >>> topology, positions = waterbox.system.topology, waterbox.positions
    >>> state = SimulationsState(temperature=298.0*unit.kelvin,
    ...                            pressure=1.0*unit.atmosphere, position=waterbox.positions)


    """

    def __init__(
        self,
        temperature: Optional[unit.Quantity] = None,
        volume: Optional[unit.Quantity] = None,
        pressure: Optional[unit.Quantity] = None,
        nr_of_particles: Optional[int] = None,
        position: Optional[jnp.ndarray] = None,
        potential: Optional[NeuralNetworkPotential] = None,
    ) -> None:
        # initialize all state variables
        self.temperature = temperature
        self.volume = volume
        self.pressure = pressure
        self.nr_of_particles = nr_of_particles
        self.position = position
        self.potential = potential

        # check which variables are not None

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
            "nr_of_particles",
            "position",
            "potential",
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

    def get_reduced_potential(self):
        """Compute the reduced potential in this thermodynamic state.

        Returns
        -------
        u : float
            The unit-less reduced potential, which can be considered
            to have units of kT.

        Raises
        ------
        ThermodynamicsError
            If the sampler state has a different number of particles.

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
        pass


class JointSimulationStates:
    """
    Manages a collection of SimulationState objects to define a joint probability distribution
    to generate samples from.
    """

    def __init__(self, states: List[SimulationState]):
        self.states = states

    def add_state(self, state: SimulationState):
        """
        Add a new SimulationState to the collection.

        Parameters
        ----------
        state : SimulationState
            The simulation state to add.
        """
        self.states.append(state)

    def check_compatibility(self):
        """
        Check the compatibility of all states in the collection.

        Returns
        -------
        bool
            True if all states are compatible, False otherwise.
        """
        for i in range(len(self.states)):
            for j in range(i + 1, len(self.states)):
                if not SimulationState.are_states_compatible(
                    self.states[i], self.states[j]
                ):
                    return False
        return True
