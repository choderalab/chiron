def test_get_list_of_mass():
    from chiron.utils import get_list_of_mass
    from openmm.app import Topology, Element

    # Create a dummy topology
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("OSC", chain)
    topology.addAtom("C", Element.getBySymbol("C"), residue)
    topology.addAtom("C", Element.getBySymbol("H"), residue)

    # Call the function
    result = get_list_of_mass(topology)

    # Check the result
    expected = [12.01, 1.01]
    assert result == expected, "Incorrect list of masses returned"
