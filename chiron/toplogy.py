from openmm.app import Topology
from typing import Dict, List, Set
import jax.numpy as jnp


# the topology class can be constructed from an openmm topology object
# the topology class needs to be able to
class Topology:
    # this class implements all possible query actions that depend on the coordinates and elements of the molecular system

    def __init__(self, topology: Topology) -> None:
        self.openmm_topology = topology

    def get_titratable_atoms(self, coordinates: jnp.array) -> Dict[str, List[int]]:
        pass

    def get_waters(self, coordinates: jnp.array) -> List[int]:
        pass

    def get_protein(self, coordinates: jnp.array) -> List[int]:
        pass

    def get_ligand_atoms(self, coordinates: jnp.array) -> List[int]:
        pass

    def get_center_of_mass(self, coordinates: jnp.array) -> jnp.array:
        pass

    def get_atomic_indices(self, coordinates: jnp.array) -> List[int]:
        pass

    def get_all_unique_elements(self, coordinates: jnp.array) -> Set[int]:
        pass

    def get_connected_protein_graph(self, coordinates: jnp.array) -> List[int]:
        pass

    def get_connected_ligand_graph(self, coordinates: jnp.array) -> List[int]:
        pass
