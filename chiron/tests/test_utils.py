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
    import pathlib

    h5_file = "langevin_reporter.h5"
    h5_test_file = get_data_file_path(h5_file)
    base_dir = (
        pathlib.Path(h5_test_file).parent.parent.absolute().joinpath("tests/data")
    )
    print(h5_test_file)
    print(base_dir)

    # Read the h5 file manually and check values
    h5 = h5py.File(h5_test_file, "r")
    keys = h5.keys()

    assert "energy" in keys, "Energy not in keys"
    assert "step" in keys, "Step not in keys"
    assert "traj" in keys, "Traj not in keys"

    energy = h5["energy"][:5]
    reference_energy = np.array(
        [1.9328993e-06, 2.0289978e-02, 8.3407544e-02, 1.7832418e-01, 2.8428176e-01]
    )
    assert np.allclose(
        energy,
        reference_energy,
    ), "Energy not correct"

    h5.close()

    # Use the reporter class and check values
    from chiron.reporters import LangevinDynamicsReporter, BaseReporter

    BaseReporter.set_directory(base_dir)

    reporter = LangevinDynamicsReporter(1)
    assert np.allclose(reference_energy, reporter.get_property("energy")[:5])
    reporter.close()
    # test the topology
    from openmmtools.testsystems import HarmonicOscillatorArray

    ho = HarmonicOscillatorArray()
    topology = ho.topology
    reporter = LangevinDynamicsReporter(1, topology)
    traj = reporter.get_mdtraj_trajectory()
    import mdtraj as md

    assert isinstance(traj, md.Trajectory), "Trajectory not correct type"
