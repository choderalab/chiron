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


import pytest
from .test_multistate import ho_multistate_sampler_multiple_ks


@pytest.fixture(scope="session")
def prep_temp_dir(tmpdir_factory):
    """Create a temporary directory for the test."""
    tmpdir = tmpdir_factory.mktemp("test_reporter")
    return tmpdir


def test_reporter(prep_temp_dir, ho_multistate_sampler_multiple_ks):
    from chiron.integrators import LangevinIntegrator
    from chiron.potential import HarmonicOscillatorPotential
    from openmm import unit

    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()
    potential = HarmonicOscillatorPotential(ho.topology)
    from chiron.utils import PRNG

    PRNG.set_seed(1234)

    from chiron.states import SamplerState, ThermodynamicState

    thermodynamic_state = ThermodynamicState(
        potential=potential, temperature=300 * unit.kelvin
    )

    sampler_state = SamplerState(ho.positions, PRNG.get_random_key())

    from chiron.reporters import LangevinDynamicsReporter
    from chiron.reporters import BaseReporter

    # set up reporter directory
    BaseReporter.set_directory(prep_temp_dir)

    # test langevin reporter
    reporter = LangevinDynamicsReporter("langevin_test")
    reporter.reset_reporter_file()

    integrator = LangevinIntegrator(
        reporter=reporter,
        report_interval=1,
    )
    integrator.run(
        sampler_state,
        thermodynamic_state,
        number_of_steps=20,
    )
    import numpy as np

    reporter.flush_buffer()

    # test for available keys
    assert "potential_energy" in reporter.get_available_keys()
    assert "step" in reporter.get_available_keys()

    # test for property
    pot_energy = reporter.get_property("potential_energy")
    np.allclose(
        pot_energy,
        np.array(
            [
                8.8336921e-05,
                3.5010747e-04,
                7.8302569e-04,
                1.4021739e-03,
                2.1981772e-03,
                3.1483083e-03,
                4.2442558e-03,
                5.4960307e-03,
                6.8922052e-03,
                8.4171966e-03,
                1.0099258e-02,
                1.1929392e-02,
                1.3859766e-02,
                1.5893064e-02,
                1.8023632e-02,
                2.0219875e-02,
                2.2491256e-02,
                2.4893485e-02,
                2.7451182e-02,
                3.0140089e-02,
            ],
            dtype=np.float32,
        ),
    )

    # test that xtc and log file is written
    import os

    assert os.path.exists(reporter.xtc_file_path)
    assert os.path.exists(reporter.log_file_path)

    # test multistate reporter
    ho_sampler = ho_multistate_sampler_multiple_ks
    ho_sampler._reporter.reset_reporter_file()
    ho_sampler.run(5)

    assert len(ho_sampler._reporter._replica_reporter.keys()) == 4
    assert ho_sampler._reporter._replica_reporter.get("replica_0")
    assert ho_sampler._reporter._default_properties == [
        "positions",
        "box_vectors",
        "u_kn",
        "state_index",
        "step",
    ]
    u_kn = ho_sampler._reporter.get_property("u_kn")
    assert u_kn.shape == (4, 4, 6)
    assert os.path.exists(ho_sampler._reporter.log_file_path)
