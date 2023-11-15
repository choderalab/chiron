class ThermodynamicState(object):
    """Thermodynamic state of a system which is defined through the following attributes:

    Attributes
    ----------
    system
    temperature
    pressure
    volume
    n_particles
    """

    @property
    def temperature(self):
        """Constant temperature of the thermodynamic state."""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value is None:
            raise RuntimeError("Cannot set temperature to None")
        self._temperature = value

    @property
    def kT(self):
        """Thermal energy per mole."""
        from openmm import unit

        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

        return kB * self.temperature

    @property
    def beta(self):
        """Thermodynamic beta in units of mole/energy."""
        return 1.0 / self.kT

    @property
    def pressure(self):
        """Constant pressure of the thermodynamic state.

        If the pressure is allowed to fluctuate, this is None. Setting
        this will automatically add/configure a barostat to the system.
        If it is set to None, the barostat will be removed.

        """
        return self._pressure

    @pressure.setter
    def pressure(self, new_pressure):
        old_pressure = self._pressure
        self._pressure = new_pressure

        # If we change ensemble, we need to modify the standard system.

    @property
    def default_box_vectors(self):
        """The default box vectors of the System (read-only)."""
        return self._standard_system.getDefaultPeriodicBoxVectors()

    @property
    def volume(self):
        """Constant volume of the thermodynamic state (read-only).

        If the volume is allowed to fluctuate, or if the system is
        not in a periodic box this is None.

        """
        return self.get_volume()

    def get_volume(self, ignore_ensemble=False):
        """Volume of the periodic box (read-only).

        Parameters
        ----------
        ignore_ensemble : bool, optional
            If True, the volume of the periodic box vectors is returned
            even if the volume fluctuates.

        Returns
        -------
        volume : openmm.unit.Quantity
            The volume of the periodic box (units of length^3) or
            None if the system is not periodic or allowed to fluctuate.

        """
        # Check if volume fluctuates
        if self.pressure is not None and not ignore_ensemble:
            return None
        if not self._standard_system.usesPeriodicBoundaryConditions():
            return None
        return _box_vectors_volume(self.default_box_vectors)

    @property
    def n_particles(self):
        """Number of particles (read-only)."""
        return self._standard_system.getNumParticles()

    @property
    def is_periodic(self):
        """True if the system is in a periodic box (read-only)."""
        return self._standard_system.usesPeriodicBoundaryConditions()


class SamplerState(object):
    """State carrying the configurational properties of a system.

    Parameters
    ----------
    positions : Nx3 unit.Quantity
        Position vectors for N particles (length units).
    velocities : Nx3 unit.Quantity, optional
        Velocity vectors for N particles (velocity units).
    box_vectors : 3x3 unit.Quantity
        Current box vectors (length units).

    Attributes
    ----------
    positions
    velocities
    box_vectors : 3x3 unit.Quantity.
        Current box vectors (length units).
    potential_energy
    kinetic_energy
    total_energy
    volume
    n_particles
    collective_variables
    """

    def __init__(self, positions, velocities=None, box_vectors=None):
        from openmm import unit
        import numpy as np
        from copy import deepcopy

        # Allocate variables, they get set in _initialize
        self._positions = None
        self._velocities = None
        self._box_vectors = None
        self._collective_variables = None
        self._kinetic_energy = None
        self._potential_energy = None
        args = []
        for input in [positions, velocities, box_vectors]:
            if isinstance(input, unit.Quantity) and not isinstance(
                input._value, np.ndarray
            ):
                args.append(np.array(input / input.unit) * input.unit)
            else:
                args.append(deepcopy(input))
        self._initialize(*args)

    @property
    def positions(self):
        """Particle positions.

        An Nx3 openmm.unit.Quantity object, where N is the number of
        particles.

        Raises
        ------
        SamplerStateError
            If set to an array with a number of particles different
            than n_particles.

        """
        return self._positions

    @positions.setter
    def positions(self, value):
        self._set_positions(value, from_context=False, check_consistency=True)

    @property
    def velocities(self):
        """Particle velocities.

        An Nx3 openmm.unit.Quantity object, where N is the number of
        particles.

        Raises
        ------
        SamplerStateError
            If set to an array with a number of particles different
            than n_particles.

        """
        return self._velocities

    @velocities.setter
    def velocities(self, value):
        self._set_velocities(value, from_context=False)

    @property
    def box_vectors(self):
        """Box vectors.

        An 3x3 openmm.unit.Quantity object.

        """
        return self._box_vectors

    @box_vectors.setter
    def box_vectors(self, value):
        from openmm import unit

        # Make sure this is a Quantity. System.getDefaultPeriodicBoxVectors
        # returns a list of Quantity objects instead for example.
        if value is not None and not isinstance(value, unit.Quantity):
            value = unit.Quantity(value)
        self._box_vectors = value
