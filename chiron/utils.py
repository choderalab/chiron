from openmm.app import Topology


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


def get_list_of_mass(topology: Topology):
    """Get the mass of the system from the topology."""
    from simtk import unit

    mass = []
    for atom in topology.atoms():
        mass.append(atom.element.mass)
    return mass
