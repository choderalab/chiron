from openmm import unit


class HarmonicOscillator:

    """Create a 3D harmonic oscillator, with a single particle confined in an isotropic harmonic well.

    Parameters
    ----------
    K : openmm.unit.Quantity, optional, default=100.0 * unit.kilocalories_per_mole/unit.angstrom**2
        harmonic restraining potential
    mass : openmm.unit.Quantity, optional, default=39.948 * unit.amu
        particle mass
    U0 : openmm.unit.Quantity, optional, default=0.0 * unit.kilocalories_per_mole
        Potential offset for harmonic oscillator

    The functional form is given by

    U(x) = (K/2) * ( (x-x0)^2 + y^2 + z^2 ) + U0

    Attributes
    ----------

    Notes
    -----

    The natural period of a harmonic oscillator is T = 2*pi*sqrt(m/K), so you will want to use an
    integration timestep smaller than ~ T/10.

    The standard deviation in position in each dimension is sigma = (kT / K)^(1/2)

    The expectation and standard deviation of the potential energy of a 3D harmonic oscillator is (3/2)kT.

    Examples
    --------

    Create a 3D harmonic oscillator with default parameters:

    >>> ho = HarmonicOscillator()
    >>> (potential, positions) = ho.system, ho.positions

    Create a harmonic oscillator with specified mass and spring constant:

    >>> mass = 12.0 * unit.amu
    >>> K = 1.0 * unit.kilocalories_per_mole / unit.angstroms**2
    >>> ho = HarmonicOscillator(K=K, mass=mass)
    >>> (system, positions) = ho.system, ho.positions

    Get a list of the available analytically-computed properties.

    >>> print(ho.analytical_properties)
    ['potential_expectation', 'potential_standard_deviation']

    Compute the potential expectation and standard deviation

    >>> import openmm.unit as u
    >>> thermodynamic_state = ThermodynamicState(temperature=298.0*u.kelvin, system=system)
    >>> potential_mean = ho.get_potential_expectation(thermodynamic_state)
    >>> potential_stddev = ho.get_potential_standard_deviation(thermodynamic_state)


    """

    def __init__(
        self,
        k=100.0 * unit.kilocalories_per_mole / unit.angstroms**2,
        U0=0.0 * unit.kilojoules_per_mole,
    ):
        from openmm.app import Topology, Element
        from .potential import HarmonicOscillatorPotential as Potential
        import numpy as np

        # Create the topology
        # Set the positions.
        positions = unit.Quantity(np.zeros([1, 3], np.float32), unit.angstroms)
        # Create topology.
        topology = Topology()
        element = Element.getBySymbol("C")
        chain = topology.addChain()
        residue = topology.addResidue("OSC", chain)
        topology.addAtom("C", element, residue)
        self.topology = topology

        # Create an empty system object.
        k = 100.0 * unit.kilocalories_per_mole / unit.angstroms**2
        U0 = 0.0 * unit.kilocalories_per_mole
        x0 = 0.0 * unit.angstrom
        harmonic_potential = Potential(k, x0, U0)

        self.K, self.U0 = k, U0
        self.harmonic_potential, self.x0 = harmonic_potential, positions

        # Number of degrees of freedom.
        self.ndof = 3

    def get_potential_expectation(self, temperature: unit.Quantity):
        """Return the expectation of the potential energy, computed analytically or numerically.

        Arguments
        ---------

        temperature : unit.Quantity.

        Returns
        -------

        potential_mean : openmm.unit.Quantity compatible with openmm.unit.kilojoules_per_mole
            The expectation of the potential energy.

        """
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

        return (3.0 / 2.0) * kB * temperature

    def get_potential_standard_deviation(self, temperature: unit.Quantity):
        """Return the standard deviation of the potential energy, computed analytically or numerically.

        Arguments
        ---------

        temperature: unit.Quantity

        Returns
        -------

        potential_stddev : openmm.unit.Quantity compatible with openmm.unit.kilojoules_per_mole
            potential energy standard deviation if implemented, or else None

        """

        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        return (3.0 / 2.0) * kB * temperature
