import pytest


@pytest.fixture
def remove_h5_file():
    import os

    if os.path.exists("test.h5"):
        os.remove("test.h5")
    yield
    if os.path.exists("test.h5"):
        os.remove("test.h5")


@pytest.fixture
def provide_testsystems_and_potentials():
    from chiron.potential import HarmonicOscillatorPotential, LJPotential
    from openmmtools.testsystems import HarmonicOscillator, LennardJonesFluid

    ho = HarmonicOscillator()
    harmonic_potential = HarmonicOscillatorPotential(ho.topology, ho.K, U0=ho.U0)

    lj_fluid = LennardJonesFluid(reduced_density=0.8, n_particles=100)
    lj_potential = LJPotential(lj_fluid.topology)

    TESTSYSTEM_AND_POTENTIAL = [
        (ho, harmonic_potential),
        (lj_fluid, lj_potential),
    ]
    return TESTSYSTEM_AND_POTENTIAL
