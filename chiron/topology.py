# Topology describes the structural organization of the particles in the system.
# It is focused on the geometrical and connectivity aspects rather than the dynamical properties.
from typing import List


class Atom:
    def __init__(self, atomic_number: int):
        from .utils import get_mass_from_atomic_number, get_symbol_from_atomic_number

        assert isinstance(atomic_number, int), "atomic_number must be an integer"
        self.atomic_number = atomic_number  # The atomic number of the atom
        self.mass = get_mass_from_atomic_number(atomic_number)  # The mass of the atom
        self.symbol = get_symbol_from_atomic_number(atomic_number)


class Topology:
    def __init__(self, atoms: List[Atom]):
        self.atoms = atoms  # atoms should be a list of Atom objects

    def __getitem__(self, index):
        # returns the Atom object at the given index
        return self.atoms[index]

    def __len__(self):
        # returns the number of atoms in the topology
        return len(self.atoms)

    def append(self, atom: Atom):
        # append an atom to the topology
        self.atoms.append(atom)

    def is_bonded(self, idx1, idx2):
        # return True if the two atoms might share molecular orbitals
        pass

    def visualize(self):
        # visualize the topology
        pass

    def visualize(self):
        # visualize the topology
        pass
