def test_get_list_of_mass():
    from chiron.utils import get_list_of_mass
    from openmm.app import Topology, Element
    from openmm import unit
    import numpy as np

    # Create a dummy topology
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("OSC", chain)
    topology.addAtom("C", Element.getBySymbol("C"), residue)
    topology.addAtom("H", Element.getBySymbol("H"), residue)

    # Call the function
    result = get_list_of_mass(topology)

    # Check the result
    expected = [12.01078, 1.01]
    c = result[0].value_in_unit_system(unit.md_unit_system)

    assert np.isclose(c, expected[0]), "Incorrect masses returned"
