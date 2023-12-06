def test_get_list_of_mass():
    """Test the get_list_of_mass function."""
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


def test_reporter():
    """Read in a reporter file and check its contend."""
    import h5py
    import numpy as np
    from chiron.utils import get_data_file_path


    h5_file = "test.h5"
    h5_test_file = get_data_file_path(h5_file)

    # Read the h5 file manually and check values
    h5 = h5py.File(h5_test_file, "r")
    keys = h5.keys()

    assert "energy" in keys, "Energy not in keys"
    assert "step" in keys, "Step not in keys"
    assert "traj" in keys, "Traj not in keys"

    energy = h5["energy"][:]
    reference_energy = np.array(
        [0.00492691, 0.08072066, 0.14170173, 0.5773072, 1.8576853]
    )
    assert np.allclose(
        energy,
        reference_energy,
    ), "Energy not correct"

    h5.close()
    
    # Use the reporter class and check values
    from chiron.reporters import SimulationReporter

    reporter = SimulationReporter(h5_test_file, 1)
    assert np.allclose(reference_energy, reporter.get_property("energy"))
