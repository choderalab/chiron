from openmm import unit
from typing import List


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

    def __init__(self) -> None:
        # initialize all state variables
        self.temperature: unit.Quantity
        self.volume: unit.Quantity
        self.pressure: unit.Quantity
        self.nr_of_particles: int
        self.position: unit.Quantity

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
    Manages a collection of SimulationState objects.
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
