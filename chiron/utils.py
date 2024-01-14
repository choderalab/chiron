from openmm.app import Topology
from openmm import unit
from jax import random


class PRNG:
    _key: random.PRNGKey
    _seed: int

    def __init__(self) -> None:
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
    from simtk import unit

    mass = []
    for atom in topology.atoms():
        mass.append(atom.element.mass.value_in_unit(unit.amu))
    return mass * unit.amu
