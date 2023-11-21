def test_generate_topology():
    from chiron.topology import Atom, Topology
    import numpy as np

    atom1, atom2, atom3, atom4 = (
        Atom(atomic_number=1),
        Atom(atomic_number=6),
        Atom(atomic_number=8),
        Atom(atomic_number=1),
    )

    topology = Topology(atoms=[atom1, atom2, atom3, atom4])
    topology.append(Atom(atomic_number=1))

    assert topology[0].atomic_number == 1
    assert np.isclose(topology[0].mass, 1.008)
    assert topology[0].symbol == "H"
    assert len(topology) == 5
