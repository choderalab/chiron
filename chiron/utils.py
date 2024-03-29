from openmm.app import Topology
from openmm import unit
from jax import random


class PRNG:
    _key: random.PRNGKey
    _seed: int

    def __init__(self) -> None:
        """
        A PRNG class that can be used to generate random numbers in JAX.
        The intended use case is to initialize new PRN streams in the `SamplerState` class.

        Example:
        --------
        from chiron.utils import PRNG
        from chiron.states import SamplerState
        from openmmtools.testsystems import HarmonicOscillator

        ho = HarmonicOscillator()
        PRNG.set_seed(1234)
        sampler_state = [SamplerState(ho.positions, PRNG.get_random_key()) for _ in x0s]

        """

        pass

    @classmethod
    def set_seed(cls, seed: int) -> None:
        cls._seed = seed
        cls._key = random.PRNGKey(seed)

    @classmethod
    def get_random_key(cls) -> int:
        key, subkey = random.split(cls._key)
        cls._key = key
        return subkey


def get_full_path(relative_path: str) -> str:
    """Get the fill path of a file that is defined relative to the chiron module root directory.

    Parameters
    ----------
    relative_path : str
        The relative path of the file.

    Returns
    -------
    str
        The full path of the file.
    """
    from importlib.resources import files

    _MODULE_ROOT = files("chiron")
    return f"{_MODULE_ROOT}/../{relative_path}"


def get_data_file_path(relative_path: str) -> str:
    """Get the full path to one of the reference files in testsystems.
    In the source distribution, these files are in ``chiron/data/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------

    name : str
        Name of the file to load (with respect to `chiron/data/`)

    """
    from importlib.resources import files

    _DATA_ROOT = files("chiron") / "data"

    file_path = _DATA_ROOT / relative_path

    if not file_path.exists():
        raise ValueError(f"Sorry! {file_path} does not exist.")

    return str(file_path)


def slice_array(arr, start_column, end_column):
    """
    Slices the array from start_column to end_column.

    Parameters:
    arr (np.ndarray): The input array.
    start_column (int): The starting column index for slicing.
    end_column (int): The ending column index for slicing (exclusive).

    Returns:
    np.ndarray: The sliced array.
    """

    return arr[:, start_column:end_column]


def get_nr_of_particles(topology: Topology) -> int:
    """Get the number of particles in the system from the topology."""
    return topology.getNumAtoms()


def get_list_of_mass(topology: Topology) -> unit.Quantity:
    """Get the mass of the system from the topology."""
    from openmm import unit

    mass = []
    for atom in topology.atoms():
        mass.append(atom.element.mass.value_in_unit(unit.amu))
    return mass * unit.amu


def initialize_velocities(
    temperature: unit.Quantity, topology: Topology, key
) -> unit.Quantity:
    """Initialize the velocities from the Maxwell-Boltzmann distribution at the given temperature.

    Parameters
    ----------
    temperature : unit.Quantity
        The temperature of the system.
    topology : Topology
        The topology of the system.
    key : int
        The PRNG key.

    """
    from openmm import unit
    import jax.numpy as jnp

    mass = get_list_of_mass(topology)

    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

    kbT_unitless = (kB * temperature).value_in_unit_system(unit.md_unit_system)
    mass_unitless = jnp.array(mass.value_in_unit_system(unit.md_unit_system))[:, None]
    sigma_v = jnp.sqrt(kbT_unitless / mass_unitless)

    v0 = sigma_v * random.normal(key, [len(mass), 3])

    return v0 * unit.nanometer / unit.picosecond
