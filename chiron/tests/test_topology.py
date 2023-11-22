from chiron.utils import get_data_file_path


def test_generate_topology():
    from openmm import app

    ethanol_pdb_file = get_data_file_path("ethanol.pdb")
    topology = app.PDBFile(ethanol_pdb_file).getTopology()
    assert topology.getNumAtoms() == 9
